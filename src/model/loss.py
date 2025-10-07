from __future__ import annotations
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SI-SDR (scale-invariant SDR) in dB.
    pred, target: [B, T]
    """
    assert pred.shape == target.shape
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = (pred * target).sum(dim=-1, keepdim=True)
    norm_t = (target**2).sum(dim=-1, keepdim=True) + eps
    s_target = dot / norm_t * target
    e_noise = pred - s_target

    num = (s_target**2).sum(dim=-1)
    den = (e_noise**2).sum(dim=-1) + eps
    ratio = num / den
    return 10 * torch.log10(ratio + eps)


class IFFARLoss(nn.Module):
    def __init__(self,
                 n_fft: int, hop_length: int, win_length: int, center_stft: bool,
                 lambda_if: float = 1.0,
                 lambda_if_reg: float = 1.0,
                 lambda_mag: float = 1.0,
                 lambda_mag_smooth: float = 0.1,
                 lambda_cons: float = 0.5,
                 lambda_stft_cons: float = 0.25,
                 lambda_overlap: float = 0.25,
                 lambda_time: float = 0.25,
                 lambda_phase: float = 0.1,
                 mag_loss: str = "huber",
                 mag_huber_delta: float = 1.0,
                 time_alpha_sisdr: float = 0.5,
                 time_beta_l1: float = 0.5,
                 cons_mode: str = "both",
                 cons_mag_only: bool = False,
                 apply_window_in_overlap: bool = True,
                 mrstft_scales: Iterable[tuple[int, int, int]] | None = None,
                 lambda_mrstft: float = 0.5,
                 lambda_if_smooth: float = 0.1,
                 if_energy_weight: float = 1.5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center_stft

        self.lambda_if = lambda_if
        self.lambda_if_reg = lambda_if_reg
        self.lambda_mag = lambda_mag
        self.lambda_mag_smooth = lambda_mag_smooth
        self.lambda_cons = lambda_cons
        self.lambda_stft_cons = lambda_stft_cons
        self.lambda_overlap = lambda_overlap
        self.lambda_time = lambda_time
        self.lambda_phase = lambda_phase

        self.mag_loss = mag_loss
        self.mag_huber_delta = mag_huber_delta
        self.time_alpha_sisdr = time_alpha_sisdr
        self.time_beta_l1 = time_beta_l1
        self.cons_mode = cons_mode
        self.cons_mag_only = cons_mag_only
        self.apply_window_in_overlap = apply_window_in_overlap
        self.mrstft_scales = list(mrstft_scales or [])
        self.lambda_mrstft = lambda_mrstft
        self.lambda_if_smooth = lambda_if_smooth
        self.if_energy_weight = if_energy_weight
        self._mrstft_eps = 1e-5
        self.register_buffer("_zero", torch.tensor(0.0))

    def _loss_mag(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mag_loss not in {"huber", "l1", "mse", "weighted_huber"}:
            raise ValueError(f"Unsupported mag_loss: {self.mag_loss}")

        if self.mag_loss == "mse":
            base = F.mse_loss(pred, target)
        elif self.mag_loss == "l1":
            base = F.l1_loss(pred, target)
        else:
            base = F.huber_loss(pred, target, delta=self.mag_huber_delta)

        if self.mag_loss == "weighted_huber":
            weight = torch.sigmoid(self.if_energy_weight * target.detach())
            loss = torch.abs(pred - target)
            mask = (loss > self.mag_huber_delta).float()
            huber = 0.5 * loss.pow(2) * (1 - mask) + (loss - 0.5 * self.mag_huber_delta) * self.mag_huber_delta * mask
            return (huber * weight).sum() / (weight.sum() + 1e-6)
        return base

    def _loss_if(self,
                 nll: torch.Tensor,
                 if_pred: torch.Tensor,
                 if_tgt: torch.Tensor,
                 mag_tgt: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        loss_nll = nll.mean()
        if mag_tgt is None:
            loss_reg = F.l1_loss(if_pred, if_tgt)
        else:
            weight = torch.sigmoid(self.if_energy_weight * mag_tgt.detach())
            loss_reg = torch.abs(if_pred - if_tgt)
            loss_reg = (loss_reg * weight).sum() / (weight.sum() + 1e-6)
        return loss_nll, loss_reg

    def _loss_if_smooth(self, if_seq: torch.Tensor | None) -> torch.Tensor:
        if if_seq is None:
            return self._zero
        seq = if_seq
        if seq.ndim == 4 and seq.size(2) == 1:
            seq = seq.squeeze(2)
        if seq.ndim != 3 or seq.size(1) < 2:
            return self._zero.to(device=seq.device)
        delta = seq[:, 1:, :] - seq[:, :-1, :]
        return torch.mean(torch.abs(delta))

    def _loss_mag_smooth(self, mag_seq: torch.Tensor | None) -> torch.Tensor:
        if mag_seq is None:
            return self._zero
        seq = mag_seq
        if seq.ndim == 2:
            return self._zero.to(device=seq.device)
        if seq.ndim == 3:
            delta = seq[:, 1:, :] - seq[:, :-1, :]
            return torch.mean(torch.abs(delta))
        return self._zero.to(device=seq.device)

    def _loss_consistency(self,
                          X_hat: torch.Tensor,
                          m_pred: torch.Tensor,
                          if_pred: torch.Tensor) -> torch.Tensor:
        mag_rec = torch.log(torch.abs(X_hat) + 1e-8)

        if mag_rec.ndim == m_pred.ndim + 1 and mag_rec.size(1) == 1:
            m_pred = m_pred.unsqueeze(1)
        elif mag_rec.ndim == m_pred.ndim and mag_rec.shape != m_pred.shape:
            raise ValueError(f"Mismatch between reconstructed magnitude {mag_rec.shape} and prediction {m_pred.shape}")

        loss_mag = F.l1_loss(mag_rec, m_pred)
        if self.cons_mag_only:
            return loss_mag
        phase_angle = torch.angle(X_hat).detach()
        if_phase_axes = phase_angle.ndim >= 3 and phase_angle.size(1) > 1
        if_pred_seq = if_pred.ndim >= 3 and if_pred.size(1) > 1
        if not (if_phase_axes and if_pred_seq):
            return loss_mag

        grad_phase = phase_angle[:, 1:] - phase_angle[:, :-1]
        grad_pred = if_pred[:, 1:] - if_pred[:, :-1]
        loss_if = F.l1_loss(grad_pred, grad_phase)
        return loss_mag + loss_if

    def _loss_stft_consistency(self, X_pred: torch.Tensor | None) -> torch.Tensor:
        if X_pred is None:
            return self._zero
        if X_pred.ndim != 3:
            return self._zero.to(device=X_pred.device)
        if X_pred.size(1) < 2:
            return self._zero.to(device=X_pred.device)
        B, K, F = X_pred.shape
        X_tf = X_pred.permute(0, 2, 1).contiguous()
        window = torch.hann_window(self.win_length, device=X_tf.device, dtype=X_tf.real.dtype)
        y = torch.istft(X_tf, n_fft=self.n_fft, hop_length=self.hop_length,
                        win_length=self.win_length, window=window,
                        center=self.center, return_complex=False)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.size(-1) == 0:
            return self._zero.to(device=X_pred.device)
        Y_tf = torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=window,
                          center=self.center, return_complex=True)
        Y_tf = Y_tf.permute(0, 2, 1)
        return torch.mean(torch.abs(Y_tf - X_pred))

    def _loss_overlap_phase(self, phi_seq: torch.Tensor, IF_seq: torch.Tensor) -> torch.Tensor:
        """
        Coerenza tra fase integrata e IF.
        phi_seq: [B,K,F], IF_seq: [B,K,F]
        """
        phi_norm = self._flatten_seq(phi_seq, squeeze_pairs=True)
        if_norm = self._flatten_seq(IF_seq)
        if phi_norm.shape != if_norm.shape:
            raise ValueError(f"Overlap loss expects matching shapes, got {phi_norm.shape} vs {if_norm.shape}")

        dphi = phi_norm[:, 1:, :] - phi_norm[:, :-1, :]
        loss = F.l1_loss(dphi, if_norm[:, 1:, :])
        return loss

    def _loss_phase_angular(self, phi_pred: torch.Tensor | None, phi_ref: torch.Tensor | None) -> torch.Tensor:
        if phi_pred is None or phi_ref is None:
            return self._zero
        phi_p = self._flatten_seq(phi_pred, squeeze_pairs=False)
        phi_r = self._flatten_seq(phi_ref, squeeze_pairs=False)
        if phi_p.shape != phi_r.shape:
            min_T = min(phi_p.size(1), phi_r.size(1))
            phi_p = phi_p[:, :min_T]
            phi_r = phi_r[:, :min_T]
        angle_diff = torch.angle(torch.exp(1j * (phi_p - phi_r)))
        return torch.mean(torch.abs(angle_diff))

    @staticmethod
    def _flatten_seq(seq: torch.Tensor, squeeze_pairs: bool = False) -> torch.Tensor:
        out = seq
        if out.ndim == 4 and out.size(2) == 1:
            out = out.squeeze(2)
        elif squeeze_pairs and out.ndim == 4 and out.size(2) == 2:
            out = out[..., 1, :] - out[..., 0, :]
        elif out.ndim >= 4:
            new_shape = out.shape[0], out.shape[1], out.shape[-1]
            out = out.reshape(new_shape)
        return out

    def _loss_mrstft(self, y_hat: torch.Tensor | None, y_ref: torch.Tensor | None) -> torch.Tensor:
        if not self.mrstft_scales or y_hat is None or y_ref is None:
            device = None
            if y_hat is not None:
                device = y_hat.device
            elif y_ref is not None:
                device = y_ref.device
            return self._zero.to(device=device) if device is not None else self._zero

        y_hat_f = y_hat.float()
        y_ref_f = y_ref.float()
        if y_hat_f.ndim == 3:
            y_hat_f = y_hat_f[:, 0, :]
        if y_ref_f.ndim == 3:
            y_ref_f = y_ref_f[:, 0, :]

        if y_hat_f.size(-1) == 0 or y_ref_f.size(-1) == 0:
            return self._zero.to(device=y_hat_f.device)
        total_sc = 0.0
        total_lsd = 0.0
        for n_fft, hop, win_length in self.mrstft_scales:
            window = torch.hann_window(win_length, device=y_hat_f.device, dtype=y_hat_f.dtype)
            Y_hat = torch.stft(y_hat_f, n_fft=n_fft, hop_length=hop, win_length=win_length,
                                window=window, center=True, return_complex=True)
            Y_ref = torch.stft(y_ref_f, n_fft=n_fft, hop_length=hop, win_length=win_length,
                                window=window, center=True, return_complex=True)

            mag_hat = torch.clamp(Y_hat.abs(), min=self._mrstft_eps)
            mag_ref = torch.clamp(Y_ref.abs(), min=self._mrstft_eps)

            lsd = torch.sqrt(((mag_hat.log() - mag_ref.log()) ** 2).mean(dim=(-2, -1)))
            total_lsd += lsd.mean()

            sc = torch.sqrt(torch.clamp(((Y_hat - Y_ref).abs() ** 2).sum(dim=(-2, -1)), min=self._mrstft_eps))
            sc = sc / (torch.sqrt(torch.clamp((Y_ref.abs() ** 2).sum(dim=(-2, -1)), min=self._mrstft_eps)))
            total_sc += sc.mean()

        num_scales = float(len(self.mrstft_scales))
        return (total_lsd + total_sc) / num_scales

    def _loss_time(self, y_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        if y_hat is None or y_ref is None:
            return torch.tensor(0.0, device=y_ref.device if y_ref is not None else "cpu")
        assert y_hat.ndim == 2 and y_ref.ndim == 2, f"Expected [B,T], got {y_hat.shape} vs {y_ref.shape}"

        y_hat_f32 = y_hat.float()
        y_ref_f32 = y_ref.float()

        sisdr = _si_sdr(y_hat_f32, y_ref_f32)
        loss_sisdr = -sisdr.mean()

        loss_l1 = F.l1_loss(y_hat_f32, y_ref_f32)

        return self.time_alpha_sisdr * loss_sisdr + self.time_beta_l1 * loss_l1

    def forward(self,
                flow_nll: torch.Tensor,
                if_pred: torch.Tensor,
                if_tgt: torch.Tensor,
                m_pred: torch.Tensor,
                m_tgt: torch.Tensor,
                X_hat: torch.Tensor,
                y_hat: torch.Tensor | None = None,
                y_ref: torch.Tensor | None = None) -> dict:

        losses = {}

        # magnitudo
        losses["mag"] = self._loss_mag(m_pred, m_tgt) * self.lambda_mag
        losses["mag_smooth"] = self._loss_mag_smooth(m_pred if m_pred.ndim == 3 else None) * self.lambda_mag_smooth

        # if
        loss_nll, loss_if_reg = self._loss_if(flow_nll, if_pred, if_tgt, m_tgt)
        losses["if_nll"] = loss_nll * self.lambda_if
        losses["if_reg"] = loss_if_reg * self.lambda_if_reg

        # consistency
        losses["cons"] = self._loss_consistency(X_hat, m_pred, if_pred) * self.lambda_cons
        losses["stft_cons"] = self._loss_stft_consistency(X_hat) * self.lambda_stft_cons

        losses["if_smooth"] = self._zero.to(device=m_pred.device)

        # time-domain
        if y_hat is not None and y_ref is not None:
            losses["time"] = self._loss_time(y_hat, y_ref) * self.lambda_time
            if self.lambda_mrstft > 0 and self.mrstft_scales:
                losses["mrstft"] = self._loss_mrstft(y_hat, y_ref) * self.lambda_mrstft
            else:
                losses["mrstft"] = torch.tensor(0.0, device=y_hat.device)
        else:
            losses["time"] = torch.tensor(0.0, device=m_pred.device)
            losses["mrstft"] = torch.tensor(0.0, device=m_pred.device)

        losses["total"] = sum(losses.values())

        return losses

    def if_smooth_penalty(self, if_seq: torch.Tensor | None) -> torch.Tensor:
        if self.lambda_if_smooth <= 0:
            return self._zero
        return self._loss_if_smooth(if_seq) * self.lambda_if_smooth

    def phase_angular_penalty(self, phi_pred: torch.Tensor | None, phi_ref: torch.Tensor | None) -> torch.Tensor:
        return self._loss_phase_angular(phi_pred, phi_ref)
