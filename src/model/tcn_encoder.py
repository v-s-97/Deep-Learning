from __future__ import annotations
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_groupnorm(num_channels: int) -> nn.GroupNorm:
    for g in (8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


class ResidualTCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation)
        self.gn1 = _make_groupnorm(d_model)
        self.gn2 = _make_groupnorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.pad(x, (self.pad, 0))
        out = self.conv1(out)
        out = F.relu(out)
        out = self.gn1(out)
        out = self.dropout(out)

        out = torch.nn.functional.pad(out, (self.pad, 0))
        out = self.conv2(out)
        out = F.relu(out)
        out = self.gn2(out)
        out = self.dropout(out)

        return x + out


class TCNContextEncoder(nn.Module):
    def __init__(self, cfg, F_bins: int = 513):
        super().__init__()
        get = (cfg.get if hasattr(cfg, "get") else lambda k, d=None: getattr(cfg, k, d))

        self.F_bins = F_bins
        self.d_model = get("d_model", 128)
        self.n_layers = get("n_layers", 4)
        self.kernel_size = get("kernel_size", 3)
        self.dropout = get("dropout", 0.1)
        self.film_dim = get("film_dim", self.d_model)
        self.n_flow_layers = get("n_flow_layers", 4)
        self.n_film_bands = get("n_film_bands", 16)

        self.input_proj = nn.Linear(2 * F_bins, self.d_model)

        blocks = []
        for i in range(self.n_layers):
            dilation = 2 ** i
            blocks.append(ResidualTCNBlock(self.d_model, self.kernel_size, dilation, self.dropout))
        self.tcn = nn.Sequential(*blocks)

        self.freq_emb = nn.Parameter(torch.zeros(F_bins, self.d_model))
        nn.init.normal_(self.freq_emb, std=0.02)

        self.film_gen = nn.Linear(self.d_model, 2 * self.n_flow_layers * self.n_film_bands * self.film_dim)

        splits = torch.linspace(0, F_bins, steps=self.n_film_bands + 1, dtype=torch.float32)
        splits = splits.round().clamp_(0, F_bins).to(torch.int64).tolist()
        self.band_slices: List[slice] = []
        for i in range(self.n_film_bands):
            s = int(splits[i]); e = int(splits[i + 1])
            self.band_slices.append(slice(s, e))

    def forward(self, M_ctx: torch.Tensor, IF_ctx: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Args:
            M_ctx, IF_ctx: [B, T, F]
        Returns:
            C: [B, F, d_model]
            film_params: lista di dict {"gamma": [B, film_dim, F], "beta": [B, film_dim, F]}
        """
        B, T, F = M_ctx.shape
        x = torch.cat([M_ctx, IF_ctx], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        h = self.tcn(x)
        h_last = h[:, :, -1]

        C = h_last.unsqueeze(1).expand(B, F, self.d_model).contiguous()
        C = C + self.freq_emb.unsqueeze(0)

        film_raw = self.film_gen(h_last)
        film_raw = film_raw.view(B, self.n_flow_layers, 2, self.n_film_bands, self.film_dim)
        film_params: List[Dict[str, torch.Tensor]] = []
        for l in range(self.n_flow_layers):
            gamma_full = torch.zeros(B, self.film_dim, self.F_bins, device=h_last.device, dtype=h_last.dtype)
            beta_full  = torch.zeros(B, self.film_dim, self.F_bins, device=h_last.device, dtype=h_last.dtype)
            for b_idx, sl in enumerate(self.band_slices):
                if sl.start == sl.stop:
                    continue
                g_band = film_raw[:, l, 0, b_idx, :]
                b_band = film_raw[:, l, 1, b_idx, :]
                width = sl.stop - sl.start
                gamma_full[:, :, sl] = g_band.unsqueeze(-1).expand(-1, -1, width)
                beta_full[:,  :, sl] = b_band.unsqueeze(-1).expand(-1, -1, width)
            film_params.append({"gamma": gamma_full, "beta": beta_full})

        return C, film_params


# Quick test
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     B, T, F = 2, 8, 513
#     cfg = {"d_model": 128, "n_layers": 4, "kernel_size": 3, "dropout": 0.1,
#            "film_dim": 64, "n_flow_layers": 4, "n_film_bands": 16}
#     enc = TCNContextEncoder(cfg, F_bins=F)
#     M_ctx = torch.randn(B, T, F)
#     IF_ctx = torch.randn(B, T, F)
#     C, film = enc(M_ctx, IF_ctx)
#     print(C.shape, film[0]["gamma"].shape, film[0]["beta"].shape)
