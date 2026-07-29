"""Pure-PyTorch scatter operations used by the Hyper-SAGNN V2E2V layers.

No ``torch_geometric`` / ``torch_scatter`` dependency: everything here is
built from ``index_add_`` / ``scatter_reduce`` so it works with a plain CPU
or CUDA PyTorch install.
"""

from __future__ import annotations

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index``.

    Args:
        src: (E, D) tensor.
        index: (E,) LongTensor in [0, dim_size).
        dim_size: number of output buckets.

    Returns:
        (dim_size, D) tensor.
    """
    shape = (dim_size,) + tuple(src.shape[1:])
    out = src.new_zeros(shape)
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean of ``src`` rows per bucket. Buckets with no entries are 0."""
    summed = scatter_sum(src, index, dim_size)
    counts = scatter_sum(torch.ones(src.shape[0], device=src.device, dtype=src.dtype), index, dim_size)
    counts = counts.clamp_min(1.0)
    shape = (dim_size,) + (1,) * (src.dim() - 1)
    return summed / counts.view(shape)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Elementwise max of ``src`` rows per bucket. Empty buckets are 0.

    Implemented with the non-inplace ``scatter_reduce`` followed by
    ``torch.where`` to avoid the "modified by an inplace operation" autograd
    error that ``scatter_reduce_`` + boolean-mask assignment triggers.
    """
    if src.dim() == 1:
        out = src.new_full((dim_size,), float("-inf"))
        out = out.scatter_reduce(0, index, src, reduce="amax", include_self=True)
    else:
        idx = index.view(-1, 1).expand_as(src)
        out = src.new_full((dim_size, src.shape[1]), float("-inf"))
        out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax of ``src`` (E,) within each bucket given by ``index`` (E,)."""
    src_max = scatter_max(src, index, dim_size)
    src = src - src_max[index]
    src_exp = src.exp()
    denom = scatter_sum(src_exp, index, dim_size)[index]
    return src_exp / denom.clamp_min(1e-16)
