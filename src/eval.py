from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import soundfile as sf  # ✅ sostituisce torchaudio.load()
import torchaudio
from torchaudio.functional import resample
import matplotlib.pyplot as plt

from dataloader import make_dataloader
from model.phase_rec import PhaseReconstructor
from train import build_model, _device_and_scaler, _load_json
from metrics import compute_metrics


# --- Config ---
CKPT_PATH = "/leonardo_work/try25_santini/Deep-Learning/checkpoints/best.pt"
BASE_DIR  = Path("/leonardo_work/try25_santini/Deep-Learning")
OUT_DIR   = BASE_DIR / "eval_out"
OUT_DIR.mkdir(exist_ok=True, parents=True)

AUDIO_DIR = OUT_DIR / "audio"
SPEC_DIR  = OUT_DIR / "spectrograms"
PLOT_DIR  = OUT_DIR / "plots"
for d in (AUDIO_DIR, SPEC_DIR, PLOT_DIR):
    d.mkdir(exist_ok=True, parents=True)

NUM_SAVE = 10


# --- Funzioni di supporto ---

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
    """Calcolo SI-SDR robusto (1D only)."""
    y_ref = y_ref.squeeze()
    y_pred = y_pred.squeeze()
    ref = y_ref - y_ref.mean()
    pred = y_pred - y_pred.mean()
    dot = torch.dot(pred, ref)
    target_energy = torch.dot(ref, ref) + eps
    proj = dot / target_energy * ref
    noise = pred - proj
    ratio = (proj.pow(2).sum() + eps) / (noise.pow(2).sum() + eps)
    return float(10 * torch.log10(ratio))


# --- Main evaluate() ---
def evaluate():
    dev, _ = _device_and_scaler()
    print(f"Using device: {dev}")
    ckpt = torch.load(CKPT_PATH, map_location=dev)
    stats = ckpt["stats"]

    # --- Manifest & setup ---
    mani_val = _load_json(str(BASE_DIR / "manifests/sr16000/val.json"))
    F_bins = int(ckpt.get("stats", {}).get("meta", {}).get("n_fft", 1024) // 2 + 1)
    L, K = 24, 4
    print(f"Detected F_bins={F_bins} (from manifest)")

    # --- Fix per i path relativi del manifest ---
    manifest_path = BASE_DIR / "manifests/sr16000/val_pairs.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        mani_pairs = json.load(f)

    if "entries" in mani_pairs:
        for e in mani_pairs["entries"]:
            for key in ["M_path", "IF_path", "PHI_path", "phi0_path"]:
                p = e.get(key)
                if p and not os.path.isabs(p):
                    abs_path = (BASE_DIR / p).resolve()
                    e[key] = str(abs_path)
                    if not abs_path.exists():
                        print(f"File non trovato -> {abs_path}", flush=True)
        fixed_manifest_path = BASE_DIR / "manifests/sr16000/val_pairs_fixed.json"
        with open(fixed_manifest_path, "w", encoding="utf-8") as f:
            json.dump(mani_pairs, f, indent=2)
    else:
        raise ValueError("Manifest malformato: manca la chiave 'entries'")

    with open(BASE_DIR / "manifests/sr16000/val.json", "r", encoding="utf-8") as f:
        val_meta = json.load(f)
    id_to_path = {entry["id"]: entry["path"] for entry in val_meta}

    hop_meta = stats.get("meta", {})
    hop = int(hop_meta.get("hop", hop_meta.get("hop_length", 256)))
    target_sr = int(hop_meta.get("sr", 16000))

    # --- Dataloader ---
    loader_val, ds_val = make_dataloader(
        manifest_path=str(fixed_manifest_path),
        L=L,
        K=K,
        batch_size=1,
        stride=K,
        num_workers=0,  # per debug
        file_shuffle=False,
        max_items_per_file=1,
        max_total_items=20,
    )
    total_samples = len(ds_val)
    print(f"Dataset length: {total_samples}", flush=True)
    print("Starting evaluation loop...", flush=True)

    # --- Modello ---
    model = build_model(F_bins, dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_metrics = []
    metrics_per_key = {}

    # --- Loop principale ---
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader_val, desc="Eval", dynamic_ncols=True, total=total_samples)):
            M_ctx  = batch["M_ctx"].to(dev, dtype=torch.float32)
            IF_ctx = batch["IF_ctx"].to(dev, dtype=torch.float32)
            M_tgt  = batch["M_tgt"].to(dev, dtype=torch.float32)
            IF_tgt = batch["IF_tgt"].to(dev, dtype=torch.float32)
            PHI_ctx = batch.get("phi_ctx")
            PHI_tgt = batch.get("phi_tgt")
            if PHI_ctx is not None:
                PHI_ctx = PHI_ctx.to(dev, dtype=torch.float32)
            if PHI_tgt is not None:
                PHI_tgt = PHI_tgt.to(dev, dtype=torch.float32)
            frame_start_tgt = int(batch.get("frame_start_tgt", torch.tensor([0]))[0].item())
            entry_id = batch.get("entry_id", [""])[0]

            # --- Percorso audio originale ---
            base_id = entry_id.split("_split")[0].split(".split")[0]
            raw_path = id_to_path.get(entry_id) or id_to_path.get(base_id)
            if raw_path is None:
                raise KeyError(f"Missing raw path for id {entry_id}")
            audio_path = Path(raw_path).expanduser()
            if not audio_path.is_absolute():
                audio_path = BASE_DIR / audio_path
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # --- Lettura audio robusta (soundfile) ---
            wav_ref_np, orig_sr = sf.read(str(audio_path), always_2d=False)
            wav_ref = torch.tensor(wav_ref_np, dtype=torch.float32)
            if wav_ref.ndim > 1:
                wav_ref = wav_ref.mean(dim=1)
            wav_ref = wav_ref.unsqueeze(0)
            if orig_sr != target_sr:
                wav_ref = resample(wav_ref, orig_sr, target_sr)
            wav_ref = wav_ref.squeeze(0)

            # --- Estrazione segmento ---
            start_sample = max(0, frame_start_tgt * hop)
            seg_len = K * hop
            if start_sample + seg_len > wav_ref.numel():
                pad = start_sample + seg_len - wav_ref.numel()
                wav_ref = torch.nn.functional.pad(wav_ref, (0, pad))
            y_ref_segment = wav_ref[start_sample:start_sample + seg_len].clone()

            # --- Forward del modello ---
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

            # --- Denormalizzazioni ---
            M_pred_denorm = PhaseReconstructor.denorm_mag(M_pred, stats)
            M_ref_denorm  = PhaseReconstructor.denorm_mag(M_tgt, stats)
            IF_pred_denorm = PhaseReconstructor.denorm_if(IF_pred, stats)
            IF_ref_denorm  = PhaseReconstructor.denorm_if(IF_tgt, stats)

            # --- Allineamento segnali ---
            y_ref_aligned, y_pred_aligned = align_signals(y_ref.squeeze(), y_pred.squeeze())

            # --- Metriche ---
            metrics = compute_metrics(
                y_ref_aligned.unsqueeze(0), y_pred_aligned.unsqueeze(0),
                M_ref_denorm.cpu(), M_pred_denorm.cpu(),
                IF_ref_denorm.cpu(), IF_pred_denorm.cpu()
            )

            # Fix: normalizza le lunghezze prima di calcolare SI-SDR
            y_ref_fix = y_ref_aligned.squeeze()
            y_pred_fix = y_pred_aligned.squeeze()
            min_len = min(y_ref_fix.numel(), y_pred_fix.numel())
            y_ref_fix = y_ref_fix[:min_len]
            y_pred_fix = y_pred_fix[:min_len]
            metrics["SI_SDR_magGT"] = compute_si_sdr(y_ref_fix, y_pred_fix)

            all_metrics.append(metrics)
            for k, v in metrics.items():
                metrics_per_key.setdefault(k, []).append(v)

            if idx < NUM_SAVE:
                tqdm.write(f"--- Debug sample {idx} ({audio_path.name}) ---")
                for k, v in metrics.items():
                    tqdm.write(f"  {k}: {v:.4f}")

            # --- Salvataggio audio ---
            ref_path = AUDIO_DIR / f"{idx:02d}_ref_{audio_path.stem}.wav"
            pred_path = AUDIO_DIR / f"{idx:02d}_pred_{audio_path.stem}.wav"
            sf.write(ref_path, y_ref_aligned.cpu().numpy(), target_sr)
            sf.write(pred_path, y_pred_aligned.cpu().numpy(), target_sr)

            # --- Spettrogrammi ---
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            mag_ref_db = 20 * np.log10(np.abs(X_ref.cpu().numpy()) + 1e-6)
            mag_pred_db = 20 * np.log10(np.abs(X_pred.cpu().numpy()) + 1e-6)
            
            mag_ref_db = mag_ref_db.squeeze(0).T
            mag_pred_db = mag_pred_db.squeeze(0).T
            diff_db = mag_pred_db - mag_ref_db

            axs[0].imshow(mag_ref_db, aspect="auto", origin="lower")
            axs[0].set_title("Reference (dB)")
            axs[1].imshow(mag_pred_db, aspect="auto", origin="lower")
            axs[1].set_title("Predicted (dB)")
            axs[2].imshow(diff_db, aspect="auto", origin="lower", cmap="bwr", vmin=-15, vmax=15)
            axs[2].set_title("Difference (Pred-Ref)")

            for ax in axs:
                ax.set_xlabel("Frame"); ax.set_ylabel("Freq bin")

            fig.tight_layout()
            fig.savefig(SPEC_DIR / f"{idx:02d}_spec_{audio_path.stem}.png", dpi=150)
            plt.close(fig)

    # --- Media finale ---
    keys = all_metrics[0].keys()
    avg_metrics = {k: float(torch.tensor([m[k] for m in all_metrics]).mean()) for k in keys}
    print("=== EVAL DONE ===")
    for k, v in avg_metrics.items():
        print(f"{k:12s}: {v:.4f}")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)

    # --- Plot metriche ---
    fig, ax = plt.subplots(figsize=(8, 4))
    metric_names = list(avg_metrics.keys())
    metric_vals = [avg_metrics[k] for k in metric_names]
    ax.bar(range(len(metric_names)), metric_vals)
    ax.set_ylabel("Value")
    ax.set_title("Average Metrics")
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "metrics_bar.png")
    plt.close(fig)


if __name__ == "__main__":
    evaluate()

