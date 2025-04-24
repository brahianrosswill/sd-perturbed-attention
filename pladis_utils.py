import torch

from typing import Optional
from comfy.ldm.modules.attention import optimized_attention


ENTMAX15_FUNC = "entmax1.5"  # sparse attention with alpha=1.5
SPARSEMAX_FUNC = "sparsemax"  # sparse attention with alpha=2


def pladis_attention_wrapper(scale: float, sparse_func: str):
    def _pladis_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        extra_options: dict,
    ):
        heads = extra_options["n_heads"]
        attn_precision = extra_options.get("attn_precision")

        # Eq. (15) implementation from PLADIS paper
        dense_result = optimized_attention(q, k, v, heads, attn_precision=attn_precision)
        sparse_result = sparse_attention(q, k, v, heads, sparse_func)
        return dense_result + scale * (sparse_result - dense_result)

    return _pladis_attention


# Simplified attention_basic with sparse functions instead of a softmax
def sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, sparse_func: str):
    b, _, dim_head = q.shape
    dim_head //= heads

    scale = dim_head**-0.5

    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    sim = q @ k.transpose(-2, -1) * scale

    del q, k

    if sparse_func == ENTMAX15_FUNC:
        sim = entmax15(sim, dim=-1)
    elif sparse_func == SPARSEMAX_FUNC:
        sim = sparsemax(sim, dim=-1)
    else:  # fallback to the default from paper
        sim = entmax15(sim, dim=-1)

    out = sim.to(v.dtype) @ v

    out = out.unsqueeze(0).reshape(b, heads, -1, dim_head).permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)
    return out


# Both sparsemax and entmax15, and all their inner functions implementations
#   are taken from https://github.com/deep-spin/entmax/blob/c2bec6d5e7d649cba7766c2172d89123ec2a6d70/entmax/activations.py
#   (as recommended by PLADIS paper)
# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>
# License: MIT
def entmax15(X: torch.Tensor, dim=-1, k: Optional[int] = None):
    max_val, _ = X.max(dim=dim, keepdim=True)
    X = X - max_val  # same numerical stability trick as for softmax
    X = X / 2  # divide by 2 to solve actual Entmax

    tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)

    Y = torch.clamp(X - tau_star, min=0) ** 2
    return Y


def sparsemax(X: torch.Tensor, dim=-1, k: Optional[int] = None):
    max_val, _ = X.max(dim=dim, keepdim=True)
    X = X - max_val  # same numerical stability trick as softmax

    tau, _ = _sparsemax_threshold_and_support(X, dim=dim, k=k)

    output = torch.clamp(X - tau, min=0)
    return output


def _entmax_threshold_and_support(X, dim=-1, k=None):
    if k is None or k >= X.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt**2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean**2)
    delta = (1 - ss) / rho

    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


def _make_ix_like(X: torch.Tensor, dim: int):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)
