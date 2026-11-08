"""Exact sparse-block helpers for Qwen3.5 checkpoints with chip layers: read only each matrix's
nonzero block, and keep cuBLAS off bf16 split-K reductions (exact_matmuls)."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


# --- weight blocks ---------------------------------------------------------------------------
#
# Most of a merged checkpoint's weight bytes are exact zeros: the chip layers use a few hundred
# of their 3584+ input dims and a fraction of their units, Qwen's widened matrices never touch
# the chip dims (bar one mute column), and the output head reads no chip dim past the output
# code. Decoding at batch 1 is bound by reading weights, so reading only each matrix's nonzero
# block is the largest exact speedup there is. The checkpoint itself stays as it is; this only
# changes which kernels run.


def exact_matmuls() -> None:
    """Make cuBLAS reduce split-K partial sums in fp32 (torch's default allows bf16).

    The circuit's GEMMs are exact only if their fp32 accumulation is: with bf16 partial sums a
    split-K kernel rounds e.g. 128 - 33280 to -33280 before adding 65536. Which kernel cuBLAS
    picks depends on the shapes (batch size, a compacted K), so every exact path sets this.
    """
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


exact_matmuls()  # importing this module makes the process's matmuls safe for the circuit


def nonzero_cols(w: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """``(w != 0).any(0)`` without materializing the whole mask (output heads are large)."""
    nz = torch.zeros(w.shape[1], dtype=torch.bool, device=w.device)
    for i in range(0, w.shape[0], chunk):
        nz |= (w[i:i + chunk] != 0).any(0)
    return nz


def _hull(mask: torch.Tensor) -> tuple[int, int]:
    idx = mask.nonzero().flatten()
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)


class SparseLinear(torch.nn.Module):
    """``F.linear(x, W)`` for a weight that is zero outside the block ``W[r0:r1, c0:c1]``;
    reads only the block. Equal to the dense product up to the order of the accumulation over
    the block's columns, so exactly equal wherever that sum is exact (all chip arithmetic)."""

    def __init__(self, w: torch.Tensor, rows=None, cols=None, block=None, dense_above=None):
        super().__init__()
        nz = w != 0 if rows is None or cols is None else None
        self.out_features, self.in_features = w.shape
        self.r0, self.r1 = rows or _hull(nz.any(1))
        self.c0, self.c1 = cols or _hull(nz.any(0))
        # dense_above: inputs of more rows than this (prefills) use the dense weight, so they
        # stay bit-identical to the stock model; the block is then a view (no extra memory)
        self.dense_above = dense_above
        self.dense = w if dense_above is not None else None
        if block is None:
            block = w[self.r0:self.r1, self.c0:self.c1]
            if dense_above is None:
                block = torch.nn.Parameter(block.contiguous(), requires_grad=False)
        self.weight = block

    def forward(self, x):
        if self.dense is not None and x.numel() > self.dense_above * x.shape[-1]:
            return F.linear(x, self.dense)
        if self.r1 == self.r0 or self.c1 == self.c0:
            return x.new_zeros(*x.shape[:-1], self.out_features)
        y = F.linear(x[..., self.c0:self.c1], self.weight)
        if self.r0 or self.r1 != self.out_features:
            y = F.pad(y, (self.r0, self.out_features - self.r1))
        return y

    def extra_repr(self):
        return (f"{self.out_features}x{self.in_features} -> rows {self.r0}:{self.r1}, "
                f"cols {self.c0}:{self.c1}")


class SparseEmbedding(torch.nn.Module):
    """Embedding whose table is zero outside columns ``c0:c1`` (shared with the output head)."""

    def __init__(self, block: torch.nn.Parameter, c0: int, dim: int):
        super().__init__()
        self.weight, self.c0, self.dim = block, c0, dim

    def forward(self, ids):
        e = F.embedding(ids, self.weight)
        return F.pad(e, (self.c0, self.dim - self.c0 - self.weight.shape[1]))


def chip_meta(model) -> dict | None:
    """The checkpoint's chipformer.json (merged checkpoints only)."""
    meta = Path(getattr(model.config, "_name_or_path", "") or ".") / "chipformer.json"
    return json.loads(meta.read_text()) if meta.exists() else None


def _n_chip_layers(model) -> int:
    meta = chip_meta(model)
    if meta is not None:
        return meta["chip_layers"]
    cfg = getattr(model.config, "text_config", model.config)
    return cfg.num_hidden_layers  # a chip-only checkpoint (tiny.py)


@torch.no_grad()
def compact(model, qwen: bool | str = True, n_chip: int | None = None) -> dict:
    """Replace the text stack's linear maps by their nonzero blocks, in place.

    ``qwen=False`` touches only the chip layers, whose arithmetic is exact, so every logit stays
    bit-identical. ``qwen="rows"`` also drops the zero output rows of Qwen's matrices (the chip
    dims they never write) in decode steps (at most 16 rows): each output is the same dot
    product, so it stays bit-identical too (dist/spikes/b19_qwenblocks.py). ``qwen=True``
    compacts Qwen's layers fully, plus the embedding and the output head: the dropped terms are
    exact zeros, but the shorter sums round differently in Qwen's own (inexact) text arithmetic,
    like any change of GEMM kernel. Returns weight counts.
    """
    exact_matmuls()
    inner = model.model
    lm = getattr(inner, "language_model", inner)
    n_chip = _n_chip_layers(model) if n_chip is None else n_chip
    before = after = 0
    heads = []
    for i, layer in enumerate(lm.layers):
        if i >= n_chip and not qwen:
            continue
        if (i < n_chip and hasattr(layer, "linear_attn")
                and isinstance(layer.linear_attn.out_proj, torch.nn.Linear)):
            heads.append(prune_heads(layer.linear_attn))
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear) and child.bias is None:
                    if i >= n_chip and qwen == "rows":
                        r0, r1 = _hull((child.weight != 0).any(1))
                        if (r0, r1) == (0, child.out_features):
                            continue
                        s = SparseLinear(child.weight, rows=(r0, r1),
                                         cols=(0, child.in_features), dense_above=16)
                    else:
                        s = SparseLinear(child.weight)
                    before += child.weight.numel()
                    after += s.weight.numel()
                    setattr(parent, name, s)
    if qwen is True and isinstance(model.lm_head, torch.nn.Linear):
        w = model.lm_head.weight
        emb = lm.embed_tokens.weight
        c0, c1 = _hull(nonzero_cols(w))
        if w.data_ptr() == emb.data_ptr():
            ce0, ce1 = _hull((emb != 0).any(0))
            c0, c1 = min(c0, ce0), max(c1, ce1)
        block = torch.nn.Parameter(w[:, c0:c1].contiguous(), requires_grad=False)
        before += w.numel()
        after += block.numel()
        if w.data_ptr() == emb.data_ptr():
            lm.embed_tokens = SparseEmbedding(block, c0, w.shape[1])
        model.lm_head = SparseLinear(w, rows=(0, w.shape[0]), cols=(c0, c1), block=block)
    torch.cuda.empty_cache()
    return {"dense_weights": before, "compact_weights": after, "chip_deltanet_heads": heads}


@torch.no_grad()
def prune_heads(la) -> tuple[int, int]:
    """Drop the value heads of a Gated DeltaNet whose output columns are all zero (with the key
    heads only they read). Heads are independent, so the kept heads compute exactly what they
    did; the recurrent state shrinks with them. Returns (kept, original) value heads."""
    nv, nk, dk, dv = la.num_v_heads, la.num_k_heads, la.head_k_dim, la.head_v_dim
    r = nv // nk
    out = la.out_proj.weight
    used = (out.view(out.shape[0], nv, dv) != 0).any(0).any(-1).view(nk, r).any(-1)
    k = used.nonzero().flatten()
    if len(k) == nk:
        return nv, nv
    if len(k) == 0:
        k = torch.zeros(1, dtype=torch.long, device=out.device)  # keep one (inert) head
    v = (k[:, None] * r + torch.arange(r, device=k.device)).flatten()
    kd = (k[:, None] * dk + torch.arange(dk, device=k.device)).flatten()
    vd = (v[:, None] * dv + torch.arange(dv, device=k.device)).flatten()
    qkv = torch.cat([kd, la.key_dim + kd, 2 * la.key_dim + vd])

    def keep(mod, idx, dim=0):
        w = mod.weight.index_select(dim, idx).contiguous()
        mod.weight = torch.nn.Parameter(w, requires_grad=False)
        if isinstance(mod, torch.nn.Linear):
            mod.out_features, mod.in_features = w.shape

    keep(la.in_proj_qkv, qkv)
    keep(la.conv1d, qkv)
    la.conv1d.in_channels = la.conv1d.out_channels = la.conv1d.groups = len(qkv)
    keep(la.in_proj_z, vd)
    keep(la.in_proj_a, v)
    keep(la.in_proj_b, v)
    keep(la.out_proj, vd, 1)
    la.A_log = torch.nn.Parameter(la.A_log[v].contiguous(), requires_grad=False)
    la.dt_bias = torch.nn.Parameter(la.dt_bias[v].contiguous(), requires_grad=False)
    la.num_k_heads, la.num_v_heads = len(k), len(v)
    la.key_dim, la.value_dim = len(k) * dk, len(v) * dv
    la.conv_dim = 2 * la.key_dim + la.value_dim
    return len(v), nv
