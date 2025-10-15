from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import torchaudio
from torchaudio.functional import resample
import matplotlib.pyplot as plt

from dataloader import make_dataloader
from model.phase_rec import PhaseReconstructor
from train import build_model, _device_and_scaler, _load_json
from metrics import compute_metrics 

CKPT_PATH = "/leonardo_work/try25_santini/Deep-Learning/checkpoints/best.pt"
OUT_DIR   = Path("eval_out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

AUDIO_DIR = OUT_DIR / "audio"
SPEC_DIR = OUT_DIR / "spectrograms"
PLOT_DIR = OUT_DIR / "plots"
for d in (AUDIO_DIR, SPEC_DIR, PLOT_DIR):
    d.mkdir(exist_ok=True, parents=True)

NUM_SAVE = 10


def align_signals(y_ref: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref_np = y_ref.detach().cpu().numpy().astype(np.float32).reshape(-1)
    pred_np = y_pred.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if ref_np.size == 0 or pred_np.size == 0:
        return y_ref.detach().cpu(), y_pred.detach().cpu()

    corr = np.correlate(pred_np, ref_np, mode="full")
    shift = int(corr.argmax()) - (len(ref_np) - 1)

    if shift > 0:
        pred_aligned = pred_np[shift:]
        ref_aligned = ref_np[:pred_aligned.shape[0]]
    elif shift < 0:
        ref_aligned = ref_np[-shift:]
        pred_aligned = pred_np[:ref_aligned.shape[0]]
    else:
        pred_aligned = pred_np
        ref_aligned = ref_np

    length = min(len(ref_aligned), len(pred_aligned))
    if length <= 0:
        length = min(len(ref_np), len(pred_np))
        pred_aligned = pred_np[:length]
        ref_aligned = ref_np[:length]

    ref_t = torch.from_numpy(ref_aligned[:length])
    pred_t = torch.from_numpy(pred_aligned[:length])
    return ref_t, pred_t


def compute_si_sdr(y_ref: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> float:
    ref = y_ref - y_ref.mean()
    pred = y_pred - y_pred.mean()
    dot = torch.dot(pred, ref)
    target_energy = torch.dot(ref, ref) + eps
    proj = dot / target_energy * ref
    noise = pred - proj
    ratio = (proj.pow(2).sum() + eps) / (noise.pow(2).sum() + eps)
    return float(10 * torch.log10(ratio))

def evaluate():
    dev, _ = _device_and_scaler()
    ckpt = torch.load(CKPT_PATH, map_location=dev)

    stats = ckpt["stats"]
    mani_val = _load_json("/leonardo_work/try25_santini/Deep-Learning/manifests/sr16000/val_pairs.json")
    F_bins, L, K = int(mani_val["F"]), 24, 8
    with open("/leonardo_work/try25_santini/Deep-Learning/manifests/sr16000/val.json", "r", encoding="utf-8") as f:
        val_meta = json.load(f)
    id_to_path = {entry["id"]: entry["path"] for entry in val_meta}
    hop_meta = stats.get("meta", {})
    hop = int(hop_meta.get("hop", hop_meta.get("hop_length", 256)))
    target_sr = int(hop_meta.get("sr", 16000))

    loader_val, ds_val = make_dataloader(
        manifest_path="/leonardo_work/try25_santini/Deep-Learning/manifests/sr16000/val_pairs.json",
        L=L,
        K=K,
        batch_size=1,
        stride=K,
        num_workers=2,
        file_shuffle=False,
        max_items_per_file=1,
    )
    total_samples = len(ds_val)

    model = build_model(F_bins, dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_metrics = []
    metrics_per_key = {}
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader_val, desc="Eval", dynamic_ncols=True, total=total_samples)):
            M_ctx  = batch["M_ctx"].to(dev, dtype=torch.float32)
            IF_ctx = batch["IF_ctx"].to(dev, dtype=torch.float32)
            M_tgt  = batch["M_tgt"].to(dev, dtype=torch.float32)   # [1,K,F]
            IF_tgt = batch["IF_tgt"].to(dev, dtype=torch.float32)
            PHI_ctx = batch.get("phi_ctx")
            PHI_tgt = batch.get("phi_tgt")
            if PHI_ctx is not None:
                PHI_ctx = PHI_ctx.to(dev, dtype=torch.float32)
            if PHI_tgt is not None:
                PHI_tgt = PHI_tgt.to(dev, dtype=torch.float32)
            frame_start_tgt = int(batch.get("frame_start_tgt", torch.tensor([0]))[0].item())
            entry_id = batch.get("entry_id", [""])[0]
            base_id = entry_id.split("_split")[0].split(".split")[0]
            raw_path = id_to_path.get(entry_id) or id_to_path.get(base_id)
            if raw_path is None:
                raise KeyError(f"Missing raw path for id {entry_id}")
            audio_path = Path(raw_path).expanduser()
            if not audio_path.is_absolute():
                audio_path = Path.cwd() / audio_path
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            wav_ref, orig_sr = torchaudio.load(str(audio_path))
            if wav_ref.dim() == 2 and wav_ref.size(0) > 1:
                wav_ref = wav_ref.mean(dim=0, keepdim=True)
            if orig_sr != target_sr:
                wav_ref = resample(wav_ref, orig_sr, target_sr)
            wav_ref = wav_ref.squeeze(0)
            start_sample = max(0, frame_start_tgt * hop)
            seg_len = K * hop
            if start_sample + seg_len > wav_ref.numel():
                pad = start_sample + seg_len - wav_ref.numel()
                wav_ref = torch.nn.functional.pad(wav_ref, (0, pad))
            y_ref_segment = wav_ref[start_sample:start_sample + seg_len].clone()

            M_pred, IF_pred, phi0_chunk = model.forward_eval(
                M_ctx, IF_ctx, stats, K=K, L=L, phi_ctx=PHI_ctx
            )
            if phi0_chunk is None:
                phi0_chunk = PHI_ctx[:, -1] if PHI_ctx is not None else None

            if not torch.isfinite(M_pred).all() or not torch.isfinite(IF_pred).all():
                raise ValueError(f"Non-finite predictions detected for entry {entry_id} at eval step {idx}")

            X_pred, phi_seq_pred, y_pred = model.recon.reconstruct_chunk(
                M_pred, IF_pred, stats, return_waveform=True, phi0=phi0_chunk
            )
            X_ref, _, _ = model.recon.reconstruct_chunk(
                M_tgt, IF_tgt, stats, return_waveform=True, phi0=phi0_chunk
            )
            y_ref = y_ref_segment

            if idx < NUM_SAVE:
                tqdm.write(f"--- Debug sample {idx} ({audio_path.name}) ---")
                M_pred_min = float(M_pred.min().cpu())
                M_pred_max = float(M_pred.max().cpu())
                M_tgt_min = float(M_tgt.min().cpu())
                M_tgt_max = float(M_tgt.max().cpu())
                tqdm.write(f"  M_pred range (norm): {M_pred_min:.3f} .. {M_pred_max:.3f}")
                tqdm.write(f"  M_tgt range (norm): {M_tgt_min:.3f} .. {M_tgt_max:.3f}")
                IF_pred_min = float(IF_pred.min().cpu())
                IF_pred_max = float(IF_pred.max().cpu())
                IF_tgt_min = float(IF_tgt.min().cpu())
                IF_tgt_max = float(IF_tgt.max().cpu())
                tqdm.write(f"  IF_pred range (norm): {IF_pred_min:.3f} .. {IF_pred_max:.3f}")
                tqdm.write(f"  IF_tgt range (norm): {IF_tgt_min:.3f} .. {IF_tgt_max:.3f}")
                X_pred_mag = X_pred.abs().mean().item()
                y_pred_std = y_pred.std().item()
                y_ref_std = y_ref.std().item()
                tqdm.write(f"  |X_pred| mean: {X_pred_mag:.6f}")
                tqdm.write(f"  y_pred std: {y_pred_std:.6f} | y_ref std: {y_ref_std:.6f}")

            M_pred_denorm = PhaseReconstructor.denorm_mag(M_pred, stats)
            M_ref_denorm = PhaseReconstructor.denorm_mag(M_tgt, stats)
            IF_pred_denorm = PhaseReconstructor.denorm_if(IF_pred, stats)
            IF_ref_denorm = PhaseReconstructor.denorm_if(IF_tgt, stats)

            y_ref_aligned, y_pred_aligned = align_signals(y_ref.squeeze(), y_pred.squeeze())
            if idx < NUM_SAVE:
                diff = (y_pred_aligned - y_ref_aligned)
                tqdm.write(f"  y_pred aligned rms: {float(y_pred_aligned.pow(2).mean().sqrt()):.6f}")
                tqdm.write(f"  y_ref aligned rms: {float(y_ref_aligned.pow(2).mean().sqrt()):.6f}")
                tqdm.write(f"  delta rms: {float(diff.pow(2).mean().sqrt()):.6f}, max|delta|: {float(diff.abs().max()):.6f}")
                tqdm.write(f"  first samples pred: {y_pred_aligned[:5].tolist()}")
                tqdm.write(f"  first samples ref : {y_ref_aligned[:5].tolist()}")

            # Ablations
            _, _, y_magref_phasepred = model.recon.reconstruct_chunk(
                M_tgt, IF_pred, stats, return_waveform=True, phi0=phi0_chunk
            )
            _, _, y_magpred_phaseref = model.recon.reconstruct_chunk(
                M_pred, IF_tgt, stats, return_waveform=True, phi0=phi0_chunk
            )
            y_ref_mag_aligned, y_magref_phasepred_aligned = align_signals(y_ref.squeeze(), y_magref_phasepred.squeeze())
            y_ref_phase_aligned, y_magpred_phaseref_aligned = align_signals(y_ref.squeeze(), y_magpred_phaseref.squeeze())
            if idx < NUM_SAVE:
                tqdm.write(f"  ablation magGT rms: {float(y_magref_phasepred_aligned.pow(2).mean().sqrt()):.6f}")
                tqdm.write(f"  ablation phaseGT rms: {float(y_magpred_phaseref_aligned.pow(2).mean().sqrt()):.6f}")

            metrics = compute_metrics(
                y_ref_aligned.unsqueeze(0), y_pred_aligned.unsqueeze(0),
                M_ref_denorm.cpu(), M_pred_denorm.cpu(),
                IF_ref_denorm.cpu(), IF_pred_denorm.cpu()
            )
            metrics["SI_SDR_phaseGT"] = compute_si_sdr(y_ref_phase_aligned, y_magpred_phaseref_aligned)
            metrics["SI_SDR_magGT"] = compute_si_sdr(y_ref_mag_aligned, y_magref_phasepred_aligned)
            all_metrics.append(metrics)
            for k, v in metrics.items():
                metrics_per_key.setdefault(k, []).append(v)

            if idx < NUM_SAVE:
                tqdm.write("  metrics:")
                for k, v in metrics.items():
                    tqdm.write(f"    {k}: {v:.4f}")
                wav_pred_path = AUDIO_DIR / f"sample_{idx:03d}_pred.wav"
                wav_ref_path = AUDIO_DIR / f"sample_{idx:03d}_ref.wav"
                def _to_2d(wav: torch.Tensor) -> torch.Tensor:
                    if wav.ndim == 1:
                        return wav.unsqueeze(0)
                    if wav.ndim == 2:
                        return wav
                    if wav.ndim == 3:
                        return wav[0]
                    raise ValueError(f"Unsupported waveform shape {wav.shape}")

                torchaudio.save(str(wav_pred_path), _to_2d(y_pred_aligned.unsqueeze(0)), 16000)
                torchaudio.save(str(wav_ref_path),  _to_2d(y_ref_aligned.unsqueeze(0)), 16000)

                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                ref_spec = M_ref_denorm[0].cpu().numpy().T
                pred_spec = M_pred_denorm[0].cpu().numpy().T
                axs[0].imshow(ref_spec, origin="lower", aspect="auto", cmap="magma")
                axs[0].set_title("Target log-mag (dB)")
                axs[1].imshow(pred_spec, origin="lower", aspect="auto", cmap="magma")
                axs[1].set_title("Pred log-mag (dB)")
                for ax in axs:
                    ax.set_xlabel("Frames")
                    ax.set_ylabel("Bins")
                fig.tight_layout()
                fig.savefig(SPEC_DIR / f"sample_{idx:03d}_spec.png")
                plt.close(fig)

    keys = all_metrics[0].keys()
    avg_metrics = {k: float(torch.tensor([m[k] for m in all_metrics]).mean()) for k in keys}
    print("=== EVAL DONE ===")
    for k,v in avg_metrics.items():
        print(f"{k:12s}: {v:.4f}")

    with open(OUT_DIR/"metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)

    # plot metriche
    fig, ax = plt.subplots(figsize=(8, 4))
    metric_names = list(avg_metrics.keys())
    metric_vals = [avg_metrics[k] for k in metric_names]
    ax.bar(range(len(metric_names)), metric_vals, color="steelblue")
    ax.set_ylabel("Value")
    ax.set_title("Average Metrics")
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "metrics_bar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, values in metrics_per_key.items():
        ax.plot(values, label=k)
    ax.set_xlabel("Validation Batch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics per sample")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "metrics_trend.png")
    plt.close(fig)

if __name__ == "__main__":
    evaluate()