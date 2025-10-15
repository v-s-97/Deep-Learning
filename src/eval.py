from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm
import torchaudio
from torchaudio.functional import resample
import matplotlib.pyplot as plt

from dataloader import make_dataloader
from model.phase_rec import PhaseReconstructor
from train import (
    build_model,
    _device_and_scaler,
    _load_json,
    CONFIG,
    _resolve_config_paths,
    _resolve_workspace_path,
)
from metrics import compute_metrics


def _resolve_path(path: str | Path) -> Path:
    return Path(_resolve_workspace_path(str(path)))


def _infer_f_from_stats(stats: dict | None) -> Optional[int]:
    if not stats or not isinstance(stats, dict):
        return None
    if "F" in stats:
        try:
            return int(stats["F"])
        except (TypeError, ValueError):
            pass
    meta = stats.get("meta")
    if isinstance(meta, dict):
        for key in ("F", "n_bins", "n_freq"):
            if key in meta:
                try:
                    return int(meta[key])
                except (TypeError, ValueError):
                    continue
        n_fft = meta.get("n_fft") or meta.get("fft") or meta.get("win_length")
        if n_fft:
            try:
                return int(n_fft) // 2 + 1
            except (TypeError, ValueError):
                pass
    return None


def _infer_f_from_manifest(manifest: dict) -> Optional[int]:
    if not isinstance(manifest, dict):
        return None
    if "F" in manifest:
        try:
            return int(manifest["F"])
        except (TypeError, ValueError):
            pass
    entries = manifest.get("entries")
    if entries:
        for entry in entries:
            if "F" in entry:
                try:
                    return int(entry["F"])
                except (TypeError, ValueError):
                    continue
    return None



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


def evaluate(
    ckpt: Union[str, Path, None] = None,
    manifest: Union[str, Path, None] = None,
    val_meta: Union[str, Path, None] = None,
    out_dir: Union[str, Path, None] = None,
    batch_size: int = 1,
    stride_override: Optional[int] = None,
    max_items_per_file: Optional[int] = 1,
    max_total_items: Optional[int] = None,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    limit: Optional[int] = None,
    num_save: int = 10,
    device: Optional[Union[str, torch.device]] = None,
    skip_audio: bool = False,
) -> dict:
    _resolve_config_paths()

    def _default_ckpt() -> Path:
        ckpt_dir = CONFIG["paths"].get("ckpt_dir") or "checkpoints"
        return Path(ckpt_dir) / "best.pt"

    manifest_default = CONFIG["paths"].get("manifest_val", "manifests/sr16000/val_pairs.json")

    ckpt_path = _resolve_path(ckpt or _default_ckpt())
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    manifest_path = _resolve_path(manifest or manifest_default)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)

    val_meta_default = manifest_path.parent / "val.json"
    val_meta_path = _resolve_path(val_meta or val_meta_default)
    id_to_path: dict[str, str] = {}
    if not skip_audio and val_meta_path.exists():
        val_meta = _load_json(val_meta_path)
        id_to_path = {entry["id"]: entry["path"] for entry in val_meta}

    out_dir = Path(out_dir) if out_dir is not None else Path("eval_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    spec_dir = out_dir / "spectrograms"
    plot_dir = out_dir / "plots"
    for d in (audio_dir, spec_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    dev, _ = _device_and_scaler()
    if device is not None:
        dev = torch.device(device)

    ckpt = torch.load(str(ckpt_path), map_location=dev)
    stats = ckpt.get("stats", {})

    F_manifest = _infer_f_from_manifest(manifest)
    F_ckpt = _infer_f_from_stats(stats)
    F_bins = F_ckpt or F_manifest
    if F_bins is None:
        raise ValueError("Unable to infer number of frequency bins from checkpoint stats or manifest.")
    if F_ckpt and F_manifest and F_ckpt != F_manifest:
        print(f"[eval] Warning: checkpoint expects F={F_ckpt}, manifest reports F={F_manifest}. Using checkpoint value.")

    L = int(CONFIG["data"]["L"])
    K = int(CONFIG["data"]["K"])
    stride = stride_override if stride_override is not None else K
    batch_size = max(1, int(batch_size))
    if max_items_per_file is not None and max_items_per_file <= 0:
        max_items = None
    else:
        max_items = max_items_per_file
    max_total = max_total_items

    num_workers = max(0, int(num_workers or 0))
    pf = prefetch_factor if num_workers > 0 else None
    if pf is not None and pf < 2:
        raise ValueError("prefetch_factor must be >= 2 when num_workers > 0.")
    pin_memory = dev.type == "cuda"

    loader_val, ds_val = make_dataloader(
        manifest_path=str(manifest_path),
        L=L,
        K=K,
        batch_size=batch_size,
        stride=stride,
        max_items_per_file=max_items,
        max_total_items=max_total,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=pf,
        persistent_workers=(num_workers > 0),
        file_shuffle=False,
        use_half=True,
        worker_seed=int(CONFIG.get("train_opts", {}).get("worker_seed", 1234)),
        drop_last=False,
    )
    total_samples = len(ds_val)
    limit_batches = limit if limit is not None else total_samples
    save_count = max(0, int(num_save))

    hop_meta = stats.get("meta", {}) if isinstance(stats, dict) else {}
    hop = int(hop_meta.get("hop") or hop_meta.get("hop_length") or CONFIG["model"]["hop"])
    target_sr = int(hop_meta.get("sr", 16000))

    model = build_model(F_bins, dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_metrics: list[dict[str, float]] = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader_val, desc="Eval", dynamic_ncols=True, total=min(limit_batches, total_samples))):
            if idx >= limit_batches:
                break

            M_ctx = batch["M_ctx"].to(dev, dtype=torch.float32)
            IF_ctx = batch["IF_ctx"].to(dev, dtype=torch.float32)
            M_tgt = batch["M_tgt"].to(dev, dtype=torch.float32)
            IF_tgt = batch["IF_tgt"].to(dev, dtype=torch.float32)
            PHI_ctx = batch.get("phi_ctx")
            PHI_tgt = batch.get("phi_tgt")
            if PHI_ctx is not None:
                PHI_ctx = PHI_ctx.to(dev, dtype=torch.float32)
            if PHI_tgt is not None:
                PHI_tgt = PHI_tgt.to(dev, dtype=torch.float32)

            frame_start_tgt = int(batch.get("frame_start_tgt", torch.tensor([0]))[0].item())
            entry_id = batch.get("entry_id", [""])[0]

            raw_path = None
            if not skip_audio and id_to_path:
                base_id = entry_id.split("_split")[0].split(".split")[0]
                raw_path = id_to_path.get(entry_id) or id_to_path.get(base_id)
                if raw_path is not None:
                    audio_path = Path(raw_path).expanduser()
                    if not audio_path.is_absolute():
                        audio_path = Path.cwd() / audio_path
                    if audio_path.exists():
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
                    else:
                        y_ref_segment = None
                else:
                    y_ref_segment = None
            else:
                y_ref_segment = None

            M_pred, IF_pred, phi0_chunk = model.forward_eval(
                M_ctx, IF_ctx, stats, K=K, L=L, phi_ctx=PHI_ctx
            )
            if phi0_chunk is None and PHI_ctx is not None:
                phi0_chunk = PHI_ctx[:, -1]

            if not torch.isfinite(M_pred).all() or not torch.isfinite(IF_pred).all():
                raise ValueError(f"Non-finite predictions detected for entry {entry_id} at eval step {idx}")

            X_pred, phi_seq_pred, y_pred = model.recon.reconstruct_chunk(
                M_pred, IF_pred, stats, return_waveform=True, phi0=phi0_chunk
            )
            X_ref, _, y_ref_seq = model.recon.reconstruct_chunk(
                M_tgt, IF_tgt, stats, return_waveform=True, phi0=phi0_chunk
            )

            y_ref_fallback = y_ref_seq.squeeze(1) if (y_ref_seq is not None and y_ref_seq.ndim == 3) else y_ref_seq
            y_ref = y_ref_segment if y_ref_segment is not None else (y_ref_fallback.squeeze(0) if isinstance(y_ref_fallback, torch.Tensor) else None)

            M_pred_denorm = PhaseReconstructor.denorm_mag(M_pred, stats)
            M_ref_denorm = PhaseReconstructor.denorm_mag(M_tgt, stats)
            IF_pred_denorm = PhaseReconstructor.denorm_if(IF_pred, stats)
            IF_ref_denorm = PhaseReconstructor.denorm_if(IF_tgt, stats)

            if y_ref is None or y_pred is None:
                continue

            y_ref_aligned, y_pred_aligned = align_signals(y_ref.squeeze(), y_pred.squeeze())

            if idx < save_count:
                tqdm.write(f"--- Debug sample {idx} ({entry_id}) ---")
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
                y_ref_std = y_ref_aligned.std().item()
                tqdm.write(f"  |X_pred| mean: {X_pred_mag:.6f}")
                tqdm.write(f"  y_pred std: {y_pred_std:.6f} | y_ref std: {y_ref_std:.6f}")

            _, _, y_magref_phasepred = model.recon.reconstruct_chunk(
                M_tgt, IF_pred, stats, return_waveform=True, phi0=phi0_chunk
            )
            _, _, y_magpred_phaseref = model.recon.reconstruct_chunk(
                M_pred, IF_tgt, stats, return_waveform=True, phi0=phi0_chunk
            )

            y_ref_mag_aligned, y_magref_phasepred_aligned = align_signals(y_ref_aligned, y_magref_phasepred.squeeze())
            y_ref_phase_aligned, y_magpred_phaseref_aligned = align_signals(y_ref_aligned, y_magpred_phaseref.squeeze())

            metrics = compute_metrics(
                y_ref_aligned.unsqueeze(0), y_pred_aligned.unsqueeze(0),
                M_ref_denorm.cpu(), M_pred_denorm.cpu(),
                IF_ref_denorm.cpu(), IF_pred_denorm.cpu()
            )
            metrics["SI_SDR_phaseGT"] = compute_si_sdr(y_ref_phase_aligned, y_magpred_phaseref_aligned)
            metrics["SI_SDR_magGT"] = compute_si_sdr(y_ref_mag_aligned, y_magref_phasepred_aligned)
            all_metrics.append(metrics)

            if idx < save_count:
                tqdm.write("  metrics:")
                for k, v in metrics.items():
                    tqdm.write(f"    {k}: {v:.4f}")

                def _to_2d(wav: torch.Tensor) -> torch.Tensor:
                    if wav.ndim == 1:
                        return wav.unsqueeze(0)
                    if wav.ndim == 2:
                        return wav
                    if wav.ndim == 3:
                        return wav[0]
                    raise ValueError(f"Unsupported waveform shape {wav.shape}")

                torchaudio.save(str(audio_dir / f"sample_{idx:03d}_pred.wav"), _to_2d(y_pred_aligned.unsqueeze(0)), target_sr)
                torchaudio.save(str(audio_dir / f"sample_{idx:03d}_ref.wav"), _to_2d(y_ref_aligned.unsqueeze(0)), target_sr)

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
                fig.savefig(spec_dir / f"sample_{idx:03d}_spec.png")
                plt.close(fig)

    if not all_metrics:
        print("[eval] No metrics computed.")
        return {}

    keys = all_metrics[0].keys()
    metrics_per_key = {k: [m[k] for m in all_metrics] for k in keys}
    avg_metrics = {k: float(torch.tensor(metrics_per_key[k]).mean()) for k in keys}
    print("=== EVAL DONE ===")
    for k, v in avg_metrics.items():
        print(f"{k:12s}: {v:.4f}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4))
    metric_names = list(avg_metrics.keys())
    metric_vals = [avg_metrics[k] for k in metric_names]
    ax.bar(range(len(metric_names)), metric_vals, color="steelblue")
    ax.set_ylabel("Value")
    ax.set_title("Average Metrics")
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plot_dir / "metrics_bar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, values in metrics_per_key.items():
        ax.plot(values, label=k)
    ax.set_xlabel("Validation Batch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics per sample")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "metrics_trend.png")
    plt.close(fig)

    return avg_metrics


if __name__ == "__main__":
    evaluate()
