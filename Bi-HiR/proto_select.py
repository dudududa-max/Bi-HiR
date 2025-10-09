
# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

# ------------------- helpers -------------------
def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device, non_blocking=True) if t.device != device else t

def _pool_queries(features_fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    features_fused: [B, C, H, W]
    returns: gap_q [B, C], gmp_q [B, C]
    """
    gap_q = F.adaptive_avg_pool2d(features_fused, 1).flatten(1)
    gmp_q = F.adaptive_max_pool2d(features_fused, 1).flatten(1)
    return gap_q, gmp_q

def _alloc_counts(K_total: int, n_gap_max: int, n_gmp_max: int) -> Tuple[int, int]:
    """
    Allocate how many to take from GAP & GMP pools for total K_total.
    Default split: floor(K/2) from GAP, remainder from GMP; borrow if a pool is short.
    """
    K_gap = min(K_total // 2, n_gap_max)
    K_gmp = min(K_total - K_gap, n_gmp_max)

    missing = K_total - (K_gap + K_gmp)
    if missing > 0:
        add_gap = min(missing, max(0, n_gap_max - K_gap))
        K_gap += add_gap
        missing -= add_gap
    if missing > 0:
        add_gmp = min(missing, max(0, n_gmp_max - K_gmp))
        K_gmp += add_gmp
        missing -= add_gmp
    return K_gap, K_gmp

def _topK_from_pool(
    query: torch.Tensor,              # [B, C]
    protos: torch.Tensor,             # [N, C]
    labels: torch.Tensor,             # [N]
    K: int,
    metric: str = "cosine",
    norm_query: bool = True,
    norm_protos: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      sims [B, K'], idx [B, K'], sel_protos [B, K', C], sel_labels [B, K']
      where K' = min(K, N). If K==0 or N==0 -> empty with correct shapes.
    """
    B = query.size(0)
    C = query.size(1)
    N = protos.size(0)

    if K <= 0 or N == 0:
        empty_sims   = query.new_zeros((B, 0))
        empty_idx    = torch.empty((B, 0), dtype=torch.long, device=query.device)
        empty_protos = query.new_zeros((B, 0, C))
        empty_labels = torch.empty((B, 0), dtype=torch.long, device=query.device)
        return empty_sims, empty_idx, empty_protos, empty_labels

    if metric == "cosine":
        q = F.normalize(query, dim=1) if norm_query else query
        p = F.normalize(protos, dim=1) if norm_protos else protos
        sim = torch.matmul(q, p.t())  # [B, N]
    elif metric == "dot":
        sim = torch.matmul(query, protos.t())  # [B, N]
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'dot'.")

    K_eff = min(K, N)
    sims, idx = torch.topk(sim, k=K_eff, dim=1, largest=True, sorted=True)  # [B, K_eff]
    flat_idx = idx.reshape(-1)                                              # [B*K_eff]
    sel_protos = protos.index_select(0, flat_idx).view(B, K_eff, C)         # [B, K_eff, C]
    sel_labels = labels.index_select(0, flat_idx).view(B, K_eff)            # [B, K_eff]
    return sims, idx, sel_protos, sel_labels


@torch.no_grad()
def query_topK_prototypes(
    features_fused: torch.Tensor,             
    gap_pool: Dict[str, torch.Tensor],         # {"prototypes":[N1,C], "labels":[N1]}
    gmp_pool: Dict[str, torch.Tensor],         # {"prototypes":[N2,C], "labels":[N2]}
    K: int,
    device: Optional[str] = None,
    metric: str = "cosine",
    normalize_queries: bool = True,
    normalize_prototypes: bool = True,
    return_similarities: bool = True,
) -> Dict[str, torch.Tensor]:

    assert K > 0, "K must be > 0"

    dev = torch.device(device) if device is not None else features_fused.device
    ff = _to_device(features_fused, dev)
    gap_q, gmp_q = _pool_queries(ff)  # [B,C], [B,C]

    gap_protos = _to_device(gap_pool["prototypes"], dev)
    gmp_protos = _to_device(gmp_pool["prototypes"], dev)
    gap_labels = _to_device(gap_pool["labels"], dev)
    gmp_labels = _to_device(gmp_pool["labels"], dev)

    n_gap = gap_protos.size(0)
    n_gmp = gmp_protos.size(0)
    K_gap, K_gmp = _alloc_counts(K, n_gap, n_gmp)

    gap_sims, gap_idx, gap_sel_p, gap_sel_y = _topK_from_pool(
        gap_q, gap_protos, gap_labels, K_gap,
        metric=metric, norm_query=normalize_queries, norm_protos=normalize_prototypes
    )
    gmp_sims, gmp_idx, gmp_sel_p, gmp_sel_y = _topK_from_pool(
        gmp_q, gmp_protos, gmp_labels, K_gmp,
        metric=metric, norm_query=normalize_queries, norm_protos=normalize_prototypes
    )

    protos = torch.cat([gap_sel_p, gmp_sel_p], dim=1)  # [B, K', C]
    labels = torch.cat([gap_sel_y, gmp_sel_y], dim=1)  # [B, K']

    B = ff.size(0)
    sources = torch.cat([
        torch.zeros((B, gap_sel_p.size(1)), dtype=torch.long, device=dev),  # 0=GAP
        torch.ones((B, gmp_sel_p.size(1)), dtype=torch.long, device=dev)    # 1=GMP
    ], dim=1)

    indices = torch.cat([
        gap_idx,
        n_gap + gmp_idx  # offset so it's index into concat([gap_protos; gmp_protos])
    ], dim=1) if protos.size(1) > 0 else torch.empty((B, 0), dtype=torch.long, device=dev)

    out = {
        "prototypes": protos,          
        "labels": labels,              
        "sources": sources,            
        "indices": indices,            
        "gap_query": gap_q,            
        "gmp_query": gmp_q,            
    }
    if return_similarities:
        out["gap_sims"] = gap_sims     
        out["gmp_sims"] = gmp_sims     
    return out



@torch.no_grad()
def query_topK_from_model(
    model,
    features_fused: torch.Tensor,   # [B, C, H, W]
    K: int,
    device: Optional[str] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Wrapper that uses pools attached by attach_dual_pools_to_model(...).
    """
    gap_pool = {"prototypes": model.prototype_pool_gap, "labels": model.prototype_pool_labels}
    gmp_pool = {"prototypes": model.prototype_pool_gmp, "labels": model.prototype_pool_labels}
    return query_topK_prototypes(features_fused, gap_pool, gmp_pool, K, device=device, **kwargs)



@torch.no_grad()
def query_topk_prototypes(*args, **kwargs):
    # allow legacy call with 'topk' kwarg
    if "topk" in kwargs and "K" not in kwargs:
        kwargs["K"] = kwargs.pop("topk")
    return query_topK_prototypes(*args, **kwargs)

@torch.no_grad()
def query_topk_from_model(*args, **kwargs):
    if "topk" in kwargs and "K" not in kwargs:
        kwargs["K"] = kwargs.pop("topk")
    return query_topK_from_model(*args, **kwargs)
