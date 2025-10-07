'''
This was the first idea for the encoder. It used a transformer. 
It was subsequently discarded due to the complexity it introduced into the overall pipeline, 
requiring long training times. It was replaced by tcn_encoder.
'''

from __future__ import annotations 
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Utility
def build_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal mask for attention over time [T, T]."""
    m = torch.full((T, T), fill_value=float('-inf'), device=device)
    return torch.triu(m, diagonal=1)


def build_band_mask(Fbins: int, band: int, device: torch.device) -> torch.Tensor:
    """Banded locality mask over frequency [F, F]: keep |i-j|<=band, -inf elsewhere."""
    i = torch.arange(Fbins, device=device)[:, None]
    j = torch.arange(Fbins, device=device)[None, :]
    dist = (i - j).abs()
    mask = torch.where(dist <= band, torch.zeros((), device=device), torch.full((), float('-inf'), device=device))
    return mask 



# Sublayers
class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.drop(self.fn(self.norm(x), *args, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalCausalMHSA(nn.Module):
    """Temporal causal attention applied per frequency bin independently.
    Input:  x [B, T, F, C]
    Output: y [B, T, F, C]
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.register_buffer('mask_cache_T', torch.zeros(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, C = x.shape
        x2 = x.permute(0, 2, 1, 3).contiguous().view(B * F, T, C)
        device = x.device
        if self.mask_cache_T.size(0) != T:
            self.mask_cache_T = build_causal_mask(T, device)
        y, _ = self.attn(x2, x2, x2, attn_mask=self.mask_cache_T)
        y = y.view(B, F, T, C).permute(0, 2, 1, 3).contiguous()
        return y


class FrequencyLocalMHSA(nn.Module):
    """Local (banded) attention across frequency at each time step (no temporal lookahead).
    Input:  x [B, T, F, C]
    Output: y [B, T, F, C]
    """
    def __init__(self, dim: int, num_heads: int = 4, band_bins: int = 16, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.band_bins = band_bins
        self.register_buffer('mask_cache_F', torch.zeros(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, C = x.shape
        x2 = x.view(B * T, F, C)
        device = x.device
        if self.mask_cache_F.size(0) != F or self.mask_cache_F.device != device:
            self.mask_cache_F = build_band_mask(F, self.band_bins, device)
        y, _ = self.attn(x2, x2, x2, attn_mask=self.mask_cache_F)
        y = y.view(B, T, F, C)
        return y


class ARContextBlock(nn.Module):
    def __init__(self, dim: int, heads_t: int, heads_f: int, band_bins: int, dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()
        self.temporal = PreNormResidual(dim, TemporalCausalMHSA(dim, heads_t, dropout), dropout)
        self.freqloc  = PreNormResidual(dim, FrequencyLocalMHSA(dim, heads_f, band_bins, dropout), dropout)
        self.ffn      = PreNormResidual(dim, FeedForward(dim, mult=ffn_mult, dropout=dropout), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x) 
        x = self.freqloc(x)  
        x = self.ffn(x)
        return x


# Main encoder
@dataclass
class ARContextEncoderConfig:
    d_model: int = 256
    n_layers: int = 6
    heads_time: int = 4
    heads_freq: int = 4
    freq_band: int = 16        
    dropout: float = 0.1
    ffn_mult: int = 4
    film_dim: int = 128        
    n_flow_layers: int = 8     


class ARContextEncoder(nn.Module):
    """
    Transformer causale compatto per IFF-AR.

    Args:
        cfg: ARContextEncoderConfig
        num_features: numero feature in input per bin (default=2: [M_norm, IF_norm])

    Input:
        M_ctx:  [B, T, F]
        IF_ctx: [B, T, F]

    Output:
        C:            [B, F, d_model]  — embedding del frame "prossimo" (usa l'ultimo hidden state a t=T-1)
        film_params:  list of dicts, len = n_flow_layers,
                      ciascuno con keys {"gamma", "beta"} shape [B, F, film_dim]
    """
    def __init__(self, cfg: ARContextEncoderConfig, num_features: int = 2):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(num_features, cfg.d_model)
        blocks = []
        for _ in range(cfg.n_layers):
            blocks.append(ARContextBlock(cfg.d_model, cfg.heads_time, cfg.heads_freq, cfg.freq_band, cfg.dropout, cfg.ffn_mult))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(cfg.d_model)

        self.film_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                nn.Linear(cfg.d_model, 2 * cfg.film_dim) 
            )
            for _ in range(cfg.n_flow_layers)
        ])

    def forward(self, M_ctx: torch.Tensor, IF_ctx: torch.Tensor) -> Tuple[torch.Tensor, List[dict]]:
        """Forward pass.
        Shapes:
            M_ctx, IF_ctx: [B, T, F]
        Returns:
            C [B, F, d_model], film_params list of dicts with gamma/beta [B, F, film_dim]
        """
        assert M_ctx.shape == IF_ctx.shape, "M_ctx and IF_ctx must have same shape [B, T, F]"
        B, T, F = M_ctx.shape
        x = torch.stack([M_ctx, IF_ctx], dim=-1)  
        x = self.in_proj(x)                       
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)

        C = x[:, -1]  

        film_params: List[dict] = []
        for mlp in self.film_mlps:
            gb = mlp(C) 
            gamma, beta = gb.split(self.cfg.film_dim, dim=-1)
            film_params.append({"gamma": gamma, "beta": beta})
        return C, film_params



# Quick test 
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     cfg = ARContextEncoderConfig()
#     enc = ARContextEncoder(cfg)
#     B, T, F = 2, 8, 513
#     M = torch.randn(B, T, F)
#     IF = torch.randn(B, T, F)
#     C, film = enc(M, IF)
#     print("C:", C.shape) 
#     print("layers:", len(film))  
#     print("gamma/beta:", film[0]["gamma"].shape, film[0]["beta"].shape)  
