# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 rickeyernest1990@gmail.com
# Licence and no-warranty terms: see LICENSE-AGPL-3.0.txt.
import math
import torch
from torch import nn
from torch.nn import functional as F


class QGSR(nn.Module):
    def __init__(self, sources, d=32, rank=4, metric_rank=8):
        super().__init__()
        self.W_O = nn.Parameter(0.1 * torch.eye(d))
        self.v = nn.Parameter(torch.zeros(sources, d))
        self.b = nn.Parameter(torch.full((sources,), math.log(4)))
        self.K = nn.Parameter(torch.randn(metric_rank, d))
        self.q = nn.Parameter(torch.zeros(d))
        self.beta_raw = nn.Parameter(torch.tensor(
            math.log(math.expm1(0.1 - 1e-5))))
        self.U = nn.Parameter(torch.randn(sources, d, rank) / math.sqrt(d))
        self.lambda_raw = nn.Parameter(torch.tensor(
            math.log(math.expm1(0.5 - 1e-3))))

    def forward(self, z, valid=None):
        if valid is None:
            valid = torch.ones(
                z.shape[:2], dtype=torch.bool, device=z.device
            )
        z = z.masked_fill(~valid[..., None], 0)
        K = F.normalize(self.K, dim=1) / math.sqrt(self.K.shape[0])
        beta = 1e-5 + F.softplus(self.beta_raw)
        lam = 1e-3 + F.softplus(self.lambda_raw)

        r = torch.sigmoid((z * self.v).sum(-1) + self.b)
        distance = F.linear(z - self.q, K).square().sum(-1)
        w = r * torch.exp(-beta * distance) * valid
        s = (w[..., None] * z).sum(1) / (1e-6 + w.sum(1, keepdim=True))

        A = w[..., None, None] * (self.U @ self.U.transpose(-1, -2))
        matrix = lam * torch.eye(
            z.shape[-1], device=z.device, dtype=z.dtype
        )
        matrix = matrix + A.sum(1)
        residual = (z - s[:, None]).unsqueeze(-1)
        rhs = (A @ residual).sum(1)
        delta = torch.linalg.solve(matrix, rhs).squeeze(-1)
        return s + F.linear(s + delta, self.W_O)
