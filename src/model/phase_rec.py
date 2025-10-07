from __future__ import annotations
import torch

class PhaseReconstructor:
    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 window: str = "hann",
                 center: bool = True,
                 return_waveform_in_chunk: bool = True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.return_waveform_in_chunk = return_waveform_in_chunk
        self.window = torch.hann_window(win_length) if window == "hann" else None

    @staticmethod
    def _broadcast_param(param, ref: torch.Tensor) -> torch.Tensor:
        out = torch.as_tensor(param, device=ref.device, dtype=ref.dtype)
        while out.ndim < ref.ndim:
            out = out.unsqueeze(0)
        return out

    @staticmethod
    def _resolve_stats(stats: dict, key: str, fallback: str) -> dict:
        entry = stats.get(key) or stats.get(fallback)
        if entry is None:
            raise KeyError(f"PhaseReconstructor stats missing '{key}'/'{fallback}' section")
        return entry

    @classmethod
    def denorm_mag(cls, M: torch.Tensor, stats: dict) -> torch.Tensor:
        """Denormalizza log-magnitudine (dB) usando mean/std globali."""
        info = cls._resolve_stats(stats, "logmag", "M")
        mean = cls._broadcast_param(info.get("mean", 0.0), M)
        std = cls._broadcast_param(info.get("std", 1.0), M)
        return M * std + mean

    @classmethod
    def denorm_if(cls, IF: torch.Tensor, stats: dict) -> torch.Tensor:
        """Denormalizza IF usando center/scale per-bin."""
        info = cls._resolve_stats(stats, "if_unwrapped", "IF")
        center = cls._broadcast_param(info.get("center", 0.0), IF)
        scale = cls._broadcast_param(info.get("scale", 1.0), IF)
        return IF * scale + center

    @staticmethod
    def integrate_phase_seq(IF_seq: torch.Tensor, phi0: torch.Tensor | None = None) -> torch.Tensor:
        """Integra IF → fase cumulativa con fase iniziale opzionale."""
        if phi0 is None:
            phi0 = torch.zeros(IF_seq.size(0), IF_seq.size(2), device=IF_seq.device, dtype=IF_seq.dtype)
        else:
            phi0 = phi0.to(device=IF_seq.device, dtype=IF_seq.dtype)
        phi0 = phi0.unsqueeze(1)
        cumsum = torch.cumsum(IF_seq, dim=1)
        return torch.cat([phi0, phi0 + cumsum], dim=1)

    @staticmethod
    def build_complex(M_db: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        M_db: [B,K,F] log-magnitudo in dB (denormalizzata)
        phi:  [B,K,F] fase cumulativa
        """
        mag = torch.pow(10.0, M_db / 20.0)
        real = mag * torch.cos(phi)
        imag = mag * torch.sin(phi)
        return torch.complex(real, imag)

    @staticmethod
    def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.ndim == 2 else x

    def reconstruct_chunk(self,
                          M_norm: torch.Tensor,
                          IF_norm: torch.Tensor,
                          stats: dict,
                          return_waveform: bool = False,
                          phi0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

        M_norm = self._ensure_sequence(M_norm).float()
        IF_norm = self._ensure_sequence(IF_norm).float()
        if M_norm.shape != IF_norm.shape:
            raise ValueError(f"Expected M/IF same shape, got {tuple(M_norm.shape)} vs {tuple(IF_norm.shape)}")

        M_denorm = self.denorm_mag(M_norm, stats)
        IF_denorm = self.denorm_if(IF_norm, stats)

        phi_seq = self.integrate_phase_seq(IF_denorm, phi0=phi0)
        X = self.build_complex(M_denorm, phi_seq[:, 1:])

        y_hat = None
        if return_waveform:
            X_istft = X.permute(0, 2, 1).contiguous()
            window = self.window
            if window is not None:
                window = window.to(device=X_istft.device, dtype=X_istft.real.dtype)
            y_hat = torch.istft(
                X_istft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=self.center,
                return_complex=False,
            )

        return X, phi_seq, y_hat

    def reconstruct(self, M: torch.Tensor, IF: torch.Tensor, stats: dict,
                    return_phase: bool = False) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        X, phi_seq, y_hat = self.reconstruct_chunk(
            M, IF, stats,
            return_waveform=self.return_waveform_in_chunk,
        )

        if self.return_waveform_in_chunk:
            if return_phase:
                return y_hat, phi_seq
            return y_hat, None

        return None, phi_seq if return_phase else None
