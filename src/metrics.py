import torch
import torch.nn.functional as F
import numpy as np

def compute_metrics(y_ref, y_pred, M_ref, M_pred, IF_ref, IF_pred):
    """
    Calcola metriche di confronto tra y_ref (waveform target) e y_pred (waveform generato).
    """
    eps = 1e-8
    
    # flatten to 1-D
    y_ref = y_ref.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Length align 
    T = min(y_ref.shape[-1], y_pred.shape[-1])
    y_ref = y_ref[..., :T]
    y_pred = y_pred[..., :T]

    if y_ref.numel() < 1024:
        pad = 1024 - y_ref.numel()
        y_ref = F.pad(y_ref, (0, pad), mode="constant", value=0.0)
        y_pred = F.pad(y_pred, (0, pad), mode="constant", value=0.0)

    y_ref = y_ref.unsqueeze(0)
    y_pred = y_pred.unsqueeze(0)

    # Log-Spectral Distance (LSD)
    window = torch.hann_window(1024, device=y_ref.device, dtype=y_ref.dtype)
    spec_ref = torch.stft(y_ref, n_fft=1024, hop_length=256, window=window, return_complex=True, center=True)
    spec_pred = torch.stft(y_pred, n_fft=1024, hop_length=256, window=window, return_complex=True, center=True)
    log_ref = torch.log(torch.abs(spec_ref) + eps)
    log_pred = torch.log(torch.abs(spec_pred) + eps)
    lsd = torch.mean(torch.sqrt(torch.mean((log_ref - log_pred) ** 2, dim=-1)))

    # Spectral Convergence
    specconv = torch.norm(spec_ref - spec_pred) / (torch.norm(spec_ref) + eps)

    # Complex MSE
    complex_mse = torch.mean(torch.abs(spec_ref - spec_pred) ** 2)

    # Magnitude metrics
    logmag_mae = F.l1_loss(M_pred, M_ref)
    logmag_rmse = torch.sqrt(F.mse_loss(M_pred, M_ref))

    # IF metrics
    if_mae = F.l1_loss(IF_pred, IF_ref)
    if_rmse = torch.sqrt(F.mse_loss(IF_pred, IF_ref))

    # SI-SDR
    def si_sdr(ref, est):
        ref_energy = torch.sum(ref ** 2) + eps
        alpha = torch.sum(est * ref) / ref_energy
        proj = alpha * ref
        noise = est - proj
        return 10 * torch.log10((torch.sum(proj ** 2) + eps) / (torch.sum(noise ** 2) + eps))

    sisdr = si_sdr(y_ref, y_pred)

    return {
        "LSD_dB": float(lsd.item()),
        "SpecConv": float(specconv.item()),
        "ComplexMSE": float(complex_mse.item()),
        "LogMag_MAE": float(logmag_mae.item()),
        "LogMag_RMSE": float(logmag_rmse.item()),
        "IF_MAE": float(if_mae.item()),
        "IF_RMSE": float(if_rmse.item()),
        "SI_SDR_dB": float(sisdr.item()),
    }
