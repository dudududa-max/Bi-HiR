import torch
import torch.nn as nn
import torch.nn.functional as F

class BiHiRLoss(nn.Module):
    def __init__(self, m_sep: float = 1.0):

        super().__init__()
        self.m_sep = m_sep

    def proto_sep_loss(self, mu: torch.Tensor) -> torch.Tensor:

        N = mu.size(0)
        if N <= 1:
            return torch.tensor(0.0, device=mu.device)
        dist = torch.cdist(mu, mu, p=2)  # [N,N]
        mask = ~torch.eye(N, dtype=bool, device=mu.device)
        margin = (self.m_sep - dist).clamp(min=0.0)
        loss = margin[mask].mean()
        return loss

    def proto_clu_loss(self, protos: torch.Tensor, labels: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:

        diffs = protos - mu[labels]  # [total,C]
        loss = (diffs.pow(2).sum(dim=1)).mean()
        return loss

    def tree_sep_loss(self, node_protos: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:

        if pairs.numel() == 0:
            return torch.tensor(0.0, device=node_protos.device)
        a = node_protos[pairs[:,0]]
        b = node_protos[pairs[:,1]]
        cos = F.cosine_similarity(a, b)
        loss = (1 - cos).mean()
        return loss

    def path_loss(self, scores: torch.Tensor) -> torch.Tensor:

        loss = 0.0
        for s in scores:  # each s: [M_i]
            loss = loss - F.log_softmax(s, dim=0).max()
        return loss

    def cls_loss(self, r_path: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        return F.nll_loss(torch.log(r_path + 1e-8), labels)

    def forward(self, mu, protos, labels, node_protos, pairs, scores, r_path, y):

        L_p_sep = self.proto_sep_loss(mu)
        L_p_clu = self.proto_clu_loss(protos, labels, mu)
        L_t_sep = self.tree_sep_loss(node_protos, pairs)
        L_path = self.path_loss(scores)
        L_cls = self.cls_loss(r_path, y)
        L_total = L_p_sep + L_p_clu + L_t_sep + L_path + L_cls
        return {
            "L_total": L_total,
            "L_p_sep": L_p_sep,
            "L_p_clu": L_p_clu,
            "L_t_sep": L_t_sep,
            "L_path": L_path,
            "L_cls": L_cls,
        }
