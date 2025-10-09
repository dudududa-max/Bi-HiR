# -*- coding: utf-8 -*-
"""
Construct separate GAP and GMP prototype pools (both match the channel dimension C of the fused feature map).
Only accepts externally provided features_fused [B, C, H, W] and labels [B]; it does NOT handle feature extraction.
"""

from typing import Dict, Iterable, Optional, Tuple
import os
import torch
import torch.nn.functional as F

# ------------------- Basic pooling -------------------
@torch.no_grad()
def _gap_gmp(features_fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:  features_fused [B, C, H, W]
    Output: gap [B, C], gmp [B, C]
    """
    gap = F.adaptive_avg_pool2d(features_fused, output_size=(1, 1)).flatten(1)
    gmp = F.adaptive_max_pool2d(features_fused, output_size=(1, 1)).flatten(1)
    return gap, gmp



@torch.no_grad()
def build_gap_gmp_pools_from_batches(
    batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: str = "cuda:0",
    normalize: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    gap_list, gmp_list, lab_list = [], [], []

    for features_fused, labels in batches:
        if not (torch.is_tensor(features_fused) and torch.is_tensor(labels)):
            raise ValueError("batches should yield a pair of tensors (features_fused, labels).")
        
        ff = features_fused.to(device, non_blocking=True)
        labs = labels.to(device, non_blocking=True)
        gap, gmp = _gap_gmp(ff)  # [B, C], [B, C]

        gap_list.append(gap.detach().cpu())
        gmp_list.append(gmp.detach().cpu())
        lab_list.append(labs.detach().cpu())

    gap_protos = torch.cat(gap_list, dim=0).to(device, non_blocking=True) if gap_list else torch.empty(0)
    gmp_protos = torch.cat(gmp_list, dim=0).to(device, non_blocking=True) if gmp_list else torch.empty(0)
    all_labels = torch.cat(lab_list, dim=0).to(device, non_blocking=True) if lab_list else torch.empty(0, dtype=torch.long)

    if normalize and gap_protos.numel() > 0:
        gap_protos = F.normalize(gap_protos, dim=1)
    if normalize and gmp_protos.numel() > 0:
        gmp_protos = F.normalize(gmp_protos, dim=1)

    gap_pool = {"prototypes": gap_protos, "labels": all_labels}
    gmp_pool = {"prototypes": gmp_protos, "labels": all_labels}
    return gap_pool, gmp_pool



@torch.no_grad()
def build_gap_gmp_pools_from_tensors(
    features_fused: torch.Tensor,  # [N, C, H, W]
    labels: torch.Tensor,          # [N]
    device: str = "cuda:0",
    normalize: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Build GAP and GMP prototype pools from full features_fused and labels tensors.
    """
    ff = features_fused.to(device, non_blocking=True)
    labs = labels.to(device, non_blocking=True)
    gap, gmp = _gap_gmp(ff)

    if normalize:
        gap = F.normalize(gap, dim=1)
        gmp = F.normalize(gmp, dim=1)

    gap_pool = {"prototypes": gap, "labels": labs}
    gmp_pool = {"prototypes": gmp, "labels": labs}
    return gap_pool, gmp_pool



def attach_dual_pools_to_model(
    model,
    gap_pool: Dict[str, torch.Tensor],
    gmp_pool: Dict[str, torch.Tensor],
) -> None:
    """
    Attach both prototype pools to a model instance so they can be used directly during training/inference.
    """
    if not hasattr(model, "prototype_pool_gap"):
        setattr(model, "prototype_pool_gap", None)
    if not hasattr(model, "prototype_pool_gmp"):
        setattr(model, "prototype_pool_gmp", None)
    if not hasattr(model, "prototype_pool_labels"):
        setattr(model, "prototype_pool_labels", None)

    model.prototype_pool_gap = gap_pool["prototypes"]
    model.prototype_pool_gmp = gmp_pool["prototypes"]

    # Labels should be identical across the two pools; use the GAP pool's labels.
    model.prototype_pool_labels = gap_pool["labels"]


def save_pool(pool: Dict[str, torch.Tensor], path: str) -> None:
    """Save a single prototype pool to disk (.pt / .pth); tensors are stored on CPU."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    torch.save(
        {"prototypes": pool["prototypes"].detach().cpu(),
         "labels": pool["labels"].detach().cpu()},
        path
    )

def load_pool(path: str, map_location: Optional[str] = "cpu") -> Dict[str, torch.Tensor]:
    """Load a single prototype pool from disk."""
    state = torch.load(path, map_location=map_location)
    if "prototypes" not in state or "labels" not in state:
        raise ValueError("File is missing 'prototypes' or 'labels'.")
    return state


__all__ = [
    "build_gap_gmp_pools_from_batches",
    "build_gap_gmp_pools_from_tensors",
    "attach_dual_pools_to_model",
    "save_pool",
    "load_pool",
]
