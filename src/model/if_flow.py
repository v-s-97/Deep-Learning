from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IFFlowConfig:
    n_layers: int = 12
    hidden: int = 256
    kernel_size: int = 7
    use_tanh_scale: bool = True
    scale_factor: float = 1.2
    eps: float = 1e-5


class CouplingNet(nn.Module):
    """
    Produce (s,t) dato x_masked e FiLM.
    Input:
        x_masked: [B,1,F]  (garantita dalla IFConditionalFlow)
        film: dict {"gamma","beta"} shape [B,F,H] o [B,H,F]
    Output:
        s, t: [B,1,F]
    """
    def __init__(self, cfg: IFFlowConfig):
        super().__init__()
        H = cfg.hidden
        k = cfg.kernel_size
        pad = k // 2
        self.conv1 = nn.Conv1d(1, H, kernel_size=k, padding=pad)
        self.conv2 = nn.Conv1d(H, H, kernel_size=k, padding=pad)
        self.out   = nn.Conv1d(H, 2, kernel_size=1)
        self.norm1 = nn.GroupNorm(num_groups=max(1, H // 16), num_channels=H)
        self.norm2 = nn.GroupNorm(num_groups=max(1, H // 16), num_channels=H)
        self.cfg = cfg

    def _prep_film(self, film: dict, Fbins: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = film["gamma"]
        beta  = film["beta"]
        if gamma.dim() != 3 or beta.dim() != 3:
            raise ValueError(f"FiLM tensors must be 3D; got gamma {gamma.shape}, beta {beta.shape}")
        if gamma.shape[-1] == self.cfg.hidden and gamma.shape[1] == Fbins:
            gamma = gamma.permute(0, 2, 1).contiguous()
            beta  = beta.permute(0, 2, 1).contiguous()
        elif gamma.shape[1] == self.cfg.hidden and gamma.shape[-1] == Fbins:
            pass
        else:
            raise ValueError(f"Unexpected FiLM shapes: gamma {gamma.shape}, expected [B,F,H] or [B,H,F] with F={Fbins}, H={self.cfg.hidden}")
        return gamma.to(device=device, dtype=dtype), beta.to(device=device, dtype=dtype)

    def forward(self, x_masked: torch.Tensor, film: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_masked.dim() == 4 and x_masked.size(1) == 1:
            x_masked = x_masked.squeeze(1)
        if x_masked.dim() != 3 or x_masked.size(1) != 1:
            raise ValueError(f"x_masked must be [B,1,F]; got {tuple(x_masked.shape)}")

        B, _, Fbins = x_masked.shape
        gamma, beta = self._prep_film(film, Fbins, x_masked.device, x_masked.dtype)

        h = self.conv1(x_masked)
        h = self.norm1(h)
        h = h * (1.0 + gamma) + beta
        h = F.relu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h * (1.0 + gamma) + beta
        h = F.relu(h)

        out = self.out(h)
        s, t = out[:, :1, :], out[:, 1:2, :]

        if self.cfg.use_tanh_scale:
            s = torch.tanh(s * self.cfg.scale_factor)
        return s, t


class IFConditionalFlow(nn.Module):
    """
    RealNVP-like 1D flow su vettori di lunghezza F (bin), condizionato via FiLM.
    Accetta y come [B,F] o [B,1,F]. Restituisce z [B,F].
    """
    def __init__(self, F_bins: int, cfg: IFFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.F = F_bins
        self.layers = nn.ModuleList([CouplingNet(cfg) for _ in range(cfg.n_layers)])
        base = torch.zeros(1, 1, F_bins)
        base[..., ::2] = 1.0
        masks = []
        for i in range(cfg.n_layers):
            masks.append(base if (i % 2 == 0) else (1.0 - base))
        self.register_buffer("masks", torch.stack(masks, dim=0))
        self.register_buffer("log2pi", torch.tensor(math.log(2.0 * math.pi), dtype=torch.float32))

    # helpers 
    def _prepare_film_list(self, film_params: List[dict]) -> List[dict]:
        if len(film_params) != len(self.layers):
            raise ValueError(f"film_list length {len(film_params)} != n_layers {len(self.layers)}")
        prepared = []
        for idx, film in enumerate(film_params):
            if not isinstance(film, dict) or "gamma" not in film or "beta" not in film:
                raise ValueError(f"Invalid film entry at layer {idx}: expected dict with 'gamma' and 'beta'")
            gamma = film["gamma"]
            beta = film["beta"]
            if gamma.dim() != 3 or beta.dim() != 3:
                raise ValueError(f"FiLM tensors must be 3D at layer {idx}, got {gamma.shape} / {beta.shape}")
            prepared.append({"gamma": gamma, "beta": beta})
        return prepared

    @staticmethod
    def _as_B1F(y: torch.Tensor) -> torch.Tensor:
        """Garantisce shape [B,1,F] da [B,F] o [B,1,F] o [B,1,1,F] (squeeze)."""
        if y.dim() == 2:
            return y.unsqueeze(1)
        if y.dim() == 3:
            if y.size(1) == 1:
                return y
        if y.dim() == 4 and y.size(1) == 1:
            return y.squeeze(1)
        raise ValueError(f"y must be [B,F] or [B,1,F]; got {tuple(y.shape)}")

    def _mask_l(self, l: int, dtype: torch.dtype) -> torch.Tensor:
        m = self.masks[l]
        if m.dim() == 4:
            m = m.squeeze(0)
        return m.to(dtype=dtype)

    # forward (y -> z)
    def f_forward(self, y: torch.Tensor, film_list: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(film_list) == len(self.layers), "film_list length must equal n_layers"
        x = self._as_B1F(y)
        B = x.size(0)
        logdet_sum = torch.zeros(B, device=x.device, dtype=torch.float32)

        for l, layer in enumerate(self.layers):
            m = self._mask_l(l, dtype=x.dtype)
            x1 = x * m
            s, t = layer(x1, film_list[l])

            xc = x * (1.0 - m)
            s32  = s.to(torch.float32)
            t32  = t.to(torch.float32)
            xc32 = xc.to(torch.float32)
            y2   = xc32 * torch.exp(s32) + t32
            x    = x1 + y2.to(x.dtype)
            logdet_sum = logdet_sum + (s32 * (1.0 - m)).sum(dim=(1, 2))

        z = x.squeeze(1)
        return z, logdet_sum

    # inverse (z -> y)
    def f_inverse(self, z: torch.Tensor, film_list: List[dict]) -> torch.Tensor:
        assert len(film_list) == len(self.layers)
        x = self._as_B1F(z)
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            m = self._mask_l(l, dtype=x.dtype)
            x1 = x * m
            s, t = layer(x1, film_list[l])

            y2  = (x - x1)
            s32 = s.to(torch.float32)
            t32 = t.to(torch.float32)
            y232 = y2.to(torch.float32)
            xc  = (y232 - t32) * torch.exp(-s32)
            x   = x1 + xc.to(x.dtype)
        return x.squeeze(1)

    def log_prob(self, y: torch.Tensor, film_list: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, logdet = self.f_forward(y, film_list)
        z32 = z.to(torch.float32)

        quad = (z32 ** 2).clamp_max(1e6).sum(dim=1) / self.F
        logp = -0.5 * (quad + self.log2pi)
        logdet_norm = logdet / self.F
        nll = -(logp + logdet_norm)

        return z, logp, logdet_norm, nll


    def sample(self, film_list: List[dict], z: Optional[torch.Tensor] = None, mean_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma0 = film_list[0]["gamma"]
        B = gamma0.size(0)
        dtype = gamma0.dtype
        device = gamma0.device
        if z is None:
            if mean_mode:
                z = torch.zeros(B, self.F, device=device, dtype=dtype)
            else:
                z = torch.randn(B, self.F, device=device, dtype=dtype)
        y = self.f_inverse(z, film_list)
        return y, z

    def forward(self, film_params: List[dict], y_target: Optional[torch.Tensor] = None, mean_mode: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Restituisce predizione IF campionata e NLL opzionale sul target."""
        film_list = self._prepare_film_list(film_params)

        if_pred, _ = self.sample(film_list, mean_mode=mean_mode)
        nll = None
        if y_target is not None:
            y_target = y_target.to(dtype=film_list[0]["gamma"].dtype)
            _, _, _, nll = self.log_prob(y_target, film_list)
        return if_pred, nll



# Quick test
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     B, Fbins = 2, 513
#     cfg = IFFlowConfig(n_layers=4, hidden=128, kernel_size=5)
#     flow = IFConditionalFlow(F_bins=Fbins, cfg=cfg)

#     film_list = []
#     for _ in range(cfg.n_layers):
#         gamma = torch.randn(B, Fbins, cfg.hidden)  # [B,F,H]
#         beta  = torch.randn(B, Fbins, cfg.hidden)
#         film_list.append({"gamma": gamma, "beta": beta})

#     # Test con y [B,F]
#     y = torch.randn(B, Fbins)
#     z, logp, logdet, nll = flow.log_prob(y, film_list)
#     print("OK [B,F]:", z.shape, logp.shape, logdet.shape, nll.shape)

#     # Test con y [B,1,F]
#     y = torch.randn(B, 1, Fbins)
#     z, logp, logdet, nll = flow.log_prob(y, film_list)
#     print("OK [B,1,F]:", z.shape, logp.shape, logdet.shape, nll.shape)

#     y_s, z_s = flow.sample(film_list)
#     print("sample y:", y_s.shape, "z:", z_s.shape)
