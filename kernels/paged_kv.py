"""Paged KV cache management for variable-length batched inference.

Manages physical pages of KV cache data. Each page holds PAGE_SIZE positions
worth of KV data for all layers and KV heads. Pages are allocated on demand
as sequences grow and freed when sequences complete.

This is the Python-side page management. The actual Triton kernels still receive
contiguous per-sequence KV buffers — the PagePool handles the conversion between
paged and contiguous representations.

Future optimization: pass page table + pool to the kernel directly to avoid
the contiguous copy. This requires modifying the attention loop to do per-tile
page table lookups.
"""

import jax.numpy as jnp
import numpy as np


class PagePool:
    """Manages a pool of physical pages for KV cache data.

    Each page stores PAGE_SIZE positions of KV data for all layers/heads.
    Pages are allocated from a fixed pool and freed back when sequences complete.

    Args:
        config: model config dict
        max_pages: total number of physical pages in the pool
        page_size: positions per page (default 64, matching KV_TILE)
    """

    def __init__(self, config, max_pages, page_size=64):
        self.page_size = page_size
        self.n_layers = config["n_layers"]
        self.n_kv_heads = config.get("n_kv_heads", config["n_heads"])
        self.d_head = config["d_head"]
        self.max_seq = config["context_len"]
        self.max_pages = max_pages
        self.max_pages_per_seq = (self.max_seq + page_size - 1) // page_size

        # Per-page element count: all layers, K+V, all KV heads, page_size positions
        self.page_elems = self.n_layers * 2 * self.n_kv_heads * page_size * self.d_head

        # Per-sequence contiguous KV size (for kernel interface)
        self.kv_per_seq = self.n_layers * 2 * self.n_kv_heads * self.max_seq * self.d_head

        # Physical page pool (CPU numpy for fast page management)
        self.pool = np.zeros((max_pages, self.page_elems), dtype=np.float16)

        # Free page tracking
        self.free_pages = list(range(max_pages))

        # Per-sequence page tables: seq_idx -> list of physical page IDs
        self.page_tables = {}

    def alloc_pages(self, n_pages):
        """Allocate n physical pages from the pool.

        Returns:
            list of physical page IDs

        Raises:
            RuntimeError if not enough free pages
        """
        if len(self.free_pages) < n_pages:
            raise RuntimeError(
                f"Page pool exhausted: need {n_pages}, have {len(self.free_pages)}")
        pages = [self.free_pages.pop() for _ in range(n_pages)]
        return pages

    def free_seq_pages(self, seq_idx):
        """Free all pages belonging to a sequence."""
        if seq_idx in self.page_tables:
            self.free_pages.extend(self.page_tables[seq_idx])
            del self.page_tables[seq_idx]

    def store_prefill_kv(self, seq_idx, k_caches, v_caches, seq_len):
        """Store prefilled KV caches into pages.

        Args:
            seq_idx: sequence identifier
            k_caches: list of (n_kv_heads, context_len, d_head) bf16 per layer
            v_caches: same
            seq_len: actual prompt length (may be < context_len)
        """
        n_pages = (seq_len + self.page_size - 1) // self.page_size
        pages = self.alloc_pages(n_pages)
        self.page_tables[seq_idx] = pages

        for page_idx, phys_page in enumerate(pages):
            pos_start = page_idx * self.page_size
            pos_end = min(pos_start + self.page_size, seq_len)

            # Pack into page layout: [layer, kv_type, head, pos, d_head]
            page_data = np.zeros(self.page_elems, dtype=np.float16)
            for layer in range(self.n_layers):
                for kv_type, caches in enumerate([k_caches, v_caches]):
                    cache_np = np.array(caches[layer])  # (n_kv_heads, ctx, d_head)
                    for head in range(self.n_kv_heads):
                        off = (layer * 2 * self.n_kv_heads * self.page_size * self.d_head
                               + kv_type * self.n_kv_heads * self.page_size * self.d_head
                               + head * self.page_size * self.d_head)
                        n_pos = pos_end - pos_start
                        page_data[off:off + n_pos * self.d_head] = (
                            cache_np[head, pos_start:pos_end, :].reshape(-1))
            self.pool[phys_page] = page_data

    def ensure_page_for_pos(self, seq_idx, position):
        """Ensure a page exists for the given position. Allocates if needed."""
        page_idx = position // self.page_size
        pages = self.page_tables.get(seq_idx, [])
        while len(pages) <= page_idx:
            new_page = self.alloc_pages(1)[0]
            pages.append(new_page)
        self.page_tables[seq_idx] = pages

    def to_contiguous(self, seq_idx):
        """Convert paged KV to contiguous format for the kernel.

        Returns:
            jnp.array of shape (kv_per_seq,) bf16, same format as pack_kv_caches()
        """
        pages = self.page_tables.get(seq_idx, [])
        kv = np.zeros(self.kv_per_seq, dtype=np.float16)

        for page_idx, phys_page in enumerate(pages):
            page_data = self.pool[phys_page]
            pos_start = page_idx * self.page_size

            for layer in range(self.n_layers):
                for kv_type in range(2):  # K=0, V=1
                    for head in range(self.n_kv_heads):
                        # Source offset in page
                        src_off = (layer * 2 * self.n_kv_heads * self.page_size * self.d_head
                                   + kv_type * self.n_kv_heads * self.page_size * self.d_head
                                   + head * self.page_size * self.d_head)
                        # Destination offset in contiguous KV
                        # Contiguous layout: [layer, kv_type, head, pos, d_head]
                        dst_off = (layer * 2 * self.n_kv_heads * self.max_seq * self.d_head
                                   + kv_type * self.n_kv_heads * self.max_seq * self.d_head
                                   + head * self.max_seq * self.d_head
                                   + pos_start * self.d_head)
                        n_elems = self.page_size * self.d_head
                        kv[dst_off:dst_off + n_elems] = page_data[src_off:src_off + n_elems]

        return jnp.array(kv, dtype=jnp.bfloat16)

    def update_from_contiguous(self, seq_idx, kv_contiguous, position):
        """Update pages from contiguous KV output (after decode step).

        Only updates the page containing the given position (the newly written position).
        """
        page_idx = position // self.page_size
        pages = self.page_tables.get(seq_idx, [])
        if page_idx >= len(pages):
            return

        phys_page = pages[page_idx]
        kv_np = np.array(kv_contiguous)
        pos_in_page = position % self.page_size

        for layer in range(self.n_layers):
            for kv_type in range(2):
                for head in range(self.n_kv_heads):
                    # Source: contiguous layout
                    src_off = (layer * 2 * self.n_kv_heads * self.max_seq * self.d_head
                               + kv_type * self.n_kv_heads * self.max_seq * self.d_head
                               + head * self.max_seq * self.d_head
                               + position * self.d_head)
                    # Destination: page layout
                    dst_off = (layer * 2 * self.n_kv_heads * self.page_size * self.d_head
                               + kv_type * self.n_kv_heads * self.page_size * self.d_head
                               + head * self.page_size * self.d_head
                               + pos_in_page * self.d_head)
                    self.pool[phys_page, dst_off:dst_off + self.d_head] = (
                        kv_np[src_off:src_off + self.d_head])

    @property
    def pages_used(self):
        return self.max_pages - len(self.free_pages)

    @property
    def memory_used_mb(self):
        return self.pages_used * self.page_elems * 2 / 1e6  # bf16 = 2 bytes

    @property
    def memory_allocated_mb(self):
        return self.max_pages * self.page_elems * 2 / 1e6
