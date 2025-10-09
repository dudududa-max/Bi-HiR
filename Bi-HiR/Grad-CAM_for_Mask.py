import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple

class GradCAMHelper:
    def __init__(self, model: nn.Module, target_layer: nn.Module):

        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.hook_a = target_layer.register_forward_hook(self._save_activation)
        self.hook_g = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):

        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):

        self.gradients = grad_out[0].detach()

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor) -> torch.Tensor:
        # Normalize CAM to [0, 1]
        cam = cam.clamp(min=0)
        cam -= cam.amin(dim=(-2, -1), keepdim=True)
        denom = cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return cam / denom

    def get_multi_cam(self, logits: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:

        A = self.activations  # [B, C, H, W]
        cams = []
        for k in range(topk_idx.size(1)):
            cls_idx = topk_idx[:, k]  # [B]

            self.model.zero_grad(set_to_none=True)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, cls_idx.view(-1, 1), 1.0)
            (logits * one_hot).sum().backward(retain_graph=True)

            G = self.gradients                              # [B, C, H, W]
            weights = G.mean(dim=(-2, -1), keepdim=True)    # [B, C, 1, 1]
            cam = (weights * A).sum(dim=1, keepdim=True)    # [B, 1, H, W]
            cams.append(self._normalize_cam(cam))


        cam_stack = torch.stack(cams, dim=1)                # [B, k, 1, H, W]
        fused_cam, _ = cam_stack.max(dim=1)                 # [B, 1, H, W]
        return fused_cam

    def close(self):

        self.hook_a.remove()
        self.hook_g.remove()


@torch.no_grad()
def cam_make_mask(
    model: nn.Module,
    target_layer: nn.Module,
    x: torch.Tensor,
    K: int = 3,
    tau_percentile: float = 50.0,
) -> torch.Tensor:
    """
    Generate and return a binary mask based on Top-k Grad-CAM.
    """
    # Need gradients to compute Grad-CAM, so keep enable_grad inside
    helper = GradCAMHelper(model, target_layer)
    with torch.enable_grad():
        logits = model(x)                         # [B, C_cls]
        probs = F.softmax(logits, dim=1)
        _, topk_idx = probs.topk(k=K, dim=1)

        cam = helper.get_multi_cam(logits, topk_idx)  # [B, 1, H, W]

    helper.close()

    # Percentile-based adaptive threshold to obtain a binary mask
    B, _, H, W = cam.shape
    cam_flat = cam.view(B, -1)
    thresh = torch.quantile(cam_flat, q=tau_percentile / 100.0, dim=1, keepdim=True)  # [B, 1]
    thresh = thresh.view(B, 1, 1, 1)
    mask = (cam >= thresh).to(cam.dtype)  # float 0/1, convenient for downstream ops

    return mask
