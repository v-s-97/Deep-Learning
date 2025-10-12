# ==============================================
# IFF-AR — Preprocessing Single-File (drop‑in, no argparse/YAML)
# Target: MacBook Pro M2 (MPS-friendly), FP16 I/O
# Run: just execute this file with Python. Edit CONFIG below as needed.
# ==============================================

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import random
import numpy as np
import soundfile as sf
import librosa
import torch

# -----------------------------
# User-editable CONFIG (Python)
# -----------------------------
CONFIG = {
    # SR & STFT
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "window": "hann",
    "center": True,
    "pad_mode": "reflect",
    "epsilon": 1.0e-7,

    # Datasets to scan (adjust paths!)
    "datasets": [
        {"name": "nsynth", "root": "/leonardo_work/try25_santini/Deep-Learning/dataset/nsynth-train/audio", "pattern": "**/*.wav"},
        # {"name": "vctk",   "root": "data/raw/vctk",   "pattern": "**/*.wav"},
    ],

    # Splits
    "splits": {"train": 0.9, "val": 0.05, "test": 0.05},
    "split_seed": 42,
    "min_duration_sec": 0.5,

    # Normalization
    # logmag: global mean/std with quantile clipping during stats
    "logmag_norm": {"type": "global", "clip_quantiles": [0.01, 0.99]},
    # if_unwrapped: per-bin median/MAD or mean/std
    "if_norm": {"type": "per_bin", "estimator": "mad", "clip_quantiles": [0.01, 0.99], "clip_value": 50.0},

    # I/O
    "io": {
        "processed_root": "data/processed",
        "manifests_root": "manifests",
        "cache_root": "data/cache_npy",
        "index_prefix": "index",
        "write_fp16": True,
    },

    # QA
    "qa": {"enable": True, "num_examples": 4, "output_dir": "qa"},
}

# =============================
# Utility I/O helpers
# =============================

def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    np.savez_compressed(path, **arrays)

# =============================
# Discovery & audio loading
# =============================

def discover_audio(root: Path, pattern: str, dataset_name: str, min_dur_sec: float = 0.5) -> List[Dict]:
    """Scan directory and collect audio files metadata.
    Returns entries: {id, path, duration, dataset}
    """
    exts = {".wav", ".flac", ".mp3"}
    paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    items = []
    for p in paths:
        try:
            info = sf.info(str(p))
            dur = float(info.frames) / float(info.samplerate)
        except Exception:
            try:
                y, sr = librosa.load(p, sr=None, mono=True)
                dur = len(y) / sr
            except Exception:
                continue
        if dur < min_dur_sec:
            continue
        items.append({
            "id": p.stem,
            "path": str(p),
            "duration": dur,
            "dataset": dataset_name,
        })
    return items


def load_audio_resample(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load mono audio in [-1, 1] float32, resampled to target_sr."""
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr
    peak = np.max(np.abs(y)) + 1e-9
    y = (y / peak).astype(np.float32)
    return y, sr

# =============================
# STFT & feature extraction
# =============================

def stft_complex(y: np.ndarray, n_fft: int, hop: int, win_length: int, window: str = "hann", center: bool = True, pad_mode: str = "reflect") -> np.ndarray:
    """Complex STFT as [T, F] (librosa returns [F, T])."""
    X = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    return X.T


def compute_logmag(mag: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Log-magnitude in dB for numpy arrays."""
    return 20.0 * np.log10(np.clip(mag, a_min=eps, a_max=None))


def compute_logmag_torch(mag: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # mag: [..., F] magnitudine lineare
    return 20.0 * torch.log10(torch.clamp(mag, min=eps))



def unwrap_phase_time(phase_tf: np.ndarray) -> np.ndarray:
    return np.unwrap(phase_tf, axis=0)


def phase_to_if(dphi_tf: np.ndarray) -> np.ndarray:
    """Instantaneous frequency as unwrapped phase derivative per hop (rad / hop)."""
    return dphi_tf.astype(np.float32)

# =============================
# Stats (global & per-bin robust)
# =============================

class RunningStats:
    """Global mean/std with optional quantile clipping during accumulation."""
    def __init__(self, clip_quantiles: Optional[Tuple[float, float]] = None):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.clip_quantiles = clip_quantiles
        self._reservoir = []
        self._reservoir_cap = 128

    def update(self, x: np.ndarray):
        if self.clip_quantiles:
            if len(self._reservoir) < self._reservoir_cap:
                flat = x.ravel()
                if flat.size > 8192:
                    flat = flat[:: max(1, flat.size // 8192)]
                self._reservoir.extend(flat.tolist())
            lo, hi = self.clip_quantiles
            arr = np.asarray(self._reservoir, dtype=np.float32)
            if arr.size > 16:
                ql, qh = np.quantile(arr, [lo, hi])
                x = np.clip(x, ql, qh)
        x = x.astype(np.float64)
        s = x.size
        self.n += s
        delta = x.mean() - self.mean
        self.mean += delta * (s / self.n)
        self.M2 += ((x - self.mean) ** 2).sum()

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.M2 / max(self.n, 1), 1e-12)))
    
    def finalize(self):
        """Ricalcola mean/std dai dati campionati nel reservoir con quantile clipping."""
        if len(self._reservoir) == 0:
            return
        arr = np.asarray(self._reservoir, dtype=np.float32)
        if self.clip_quantiles:
            ql, qh = np.quantile(arr, self.clip_quantiles)
            arr = np.clip(arr, ql, qh)
        self.mean = float(arr.mean())
        self.M2 = float(arr.var() * arr.size)
        self.n = arr.size


class RobustPerBin:
    """Per-frequency robust center/scale using median & MAD (or mean/std) with optional quantile clipping."""
    def __init__(self, n_bins: int, estimator: str = "mad", clip_quantiles: Optional[Tuple[float, float]] = None):
        self.n_bins = n_bins
        self.estimator = estimator
        self.clip_quantiles = clip_quantiles
        self._med_buffers: List[np.ndarray] = []
        self._cap = 64
        self._sum = np.zeros((n_bins,), dtype=np.float64)
        self._sum2 = np.zeros((n_bins,), dtype=np.float64)
        self._count = np.zeros((n_bins,), dtype=np.int64)

    def update(self, IF: np.ndarray):
        T, F = IF.shape
        assert F == self.n_bins
        X = IF
        if self.clip_quantiles:
            lo, hi = self.clip_quantiles
            flat = X.ravel()
            if flat.size > 32768:
                flat = flat[:: max(1, flat.size // 32768)]
            ql, qh = np.quantile(flat, [lo, hi])
            X = np.clip(X, ql, qh)
        if self.estimator == "mad":
            if len(self._med_buffers) < self._cap:
                step = max(1, T // 512)
                self._med_buffers.append(X[::step].astype(np.float32))
        else:
            self._sum += X.sum(axis=0)
            self._sum2 += (X ** 2).sum(axis=0)
            self._count += T

    def to_dict(self) -> Dict:
        if self.estimator == "mad":
            if len(self._med_buffers) == 0:
                raise RuntimeError("No data for median/MAD stats.")
            Z = np.concatenate(self._med_buffers, axis=0)
            median = np.median(Z, axis=0)
            mad = np.median(np.abs(Z - median[None, :]), axis=0)
            scale = np.maximum(mad * 1.4826, 1e-6)
            return {
                "type": "per_bin_mad",
                "center": median.astype(np.float32).tolist(),
                "scale": scale.astype(np.float32).tolist(),
                "clip_quantiles": self.clip_quantiles,
            }
        else:
            mean = self._sum / np.maximum(self._count, 1)
            var = self._sum2 / np.maximum(self._count, 1) - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-12))
            return {
                "type": "per_bin_meanstd",
                "center": mean.astype(np.float32).tolist(),
                "scale": std.astype(np.float32).tolist(),
                "clip_quantiles": self.clip_quantiles,
            }

# =============================
# Splitting
# =============================

def make_splits(items: List[Dict], split_fracs: Dict[str, float], seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    items_shuf = items[:]
    rng.shuffle(items_shuf)
    n = len(items_shuf)
    n_train = int(split_fracs.get("train", 0.9) * n)
    n_val = int(split_fracs.get("val", 0.05) * n)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    for it, sp in zip(items_shuf, splits):
        it["split"] = sp
    return items_shuf

# =============================
# Passes & normalization
# =============================

def pass1_collect_stats(cfg: dict, manifest: List[Dict]) -> dict:
    """
    Primo passaggio: colleziona statistiche globali su log-magnitudine e IF
    (train set soltanto).
    - logmag: mean/std globali con clipping ai quantili.
    - IF: per-bin median/MAD o mean/std (configurabile).
    """
    sr = cfg["sample_rate"]
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    win_length = cfg["win_length"]
    eps = float(cfg["epsilon"])

    # --- inizializza collector ---
    logmag_stats = RunningStats(
        clip_quantiles=tuple(cfg["logmag_norm"].get("clip_quantiles", []))
        if cfg["logmag_norm"].get("clip_quantiles")
        else None
    )
    if_cfg = cfg["if_norm"]
    F_bins = n_fft // 2 + 1
    if_stats = RobustPerBin(
        n_bins=F_bins,
        estimator=if_cfg.get("estimator", "mad"),
        clip_quantiles=tuple(if_cfg.get("clip_quantiles", []))
        if if_cfg.get("clip_quantiles")
        else None,
    )

    # --- loop sui file di train ---
    for ex in manifest:
        if ex.get("split") != "train":
            continue

        # carica e STFT
        y, _ = load_audio_resample(ex["path"], sr)
        X = stft_complex(
            y,
            n_fft=n_fft,
            hop=hop,
            win_length=win_length,
            window=cfg["window"],
            center=cfg["center"],
            pad_mode=cfg["pad_mode"],
        )  # [T,F]

        mag = np.abs(X)
        logmag = compute_logmag(mag, eps=eps)
        phase = np.angle(X)
        phase_u = unwrap_phase_time(phase)
        dphi = phase_u[1:] - phase_u[:-1]
        IF = phase_to_if(dphi)

        # allinea a IF (T-1)
        logmag = logmag[1:]

        # accumula stats
        logmag_stats.update(logmag)
        if_stats.update(IF)

    # --- ricalcola mean/std finali da reservoir con clipping ---
    logmag_stats.finalize()

    # --- costruisci dizionario stats ---
    stats = {
        "logmag": {
            "type": "global",
            "mean": float(logmag_stats.mean),
            "std": float(logmag_stats.std),
            "clip_quantiles": cfg["logmag_norm"].get("clip_quantiles"),
        },
        "if_unwrapped": if_stats.to_dict(),
        "meta": {"sr": sr, "n_fft": n_fft, "hop": hop, "win_length": win_length},
    }
    return stats



def normalize_features(cfg: dict, stats: dict, logmag: np.ndarray, IF: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lm = (logmag - stats["logmag"]["mean"]) / max(stats["logmag"]["std"], 1e-6)
    info = stats["if_unwrapped"]
    center = np.asarray(info["center"], dtype=np.float32)
    scale = np.asarray(info["scale"], dtype=np.float32)
    IFn = (IF - center[None, :]) / np.maximum(scale[None, :], 1e-6)
    clip_val = cfg.get("if_norm", {}).get("clip_value")
    if clip_val is None:
        clip_val = cfg.get("if_norm_clip")
    if clip_val is not None:
        IFn = np.clip(IFn, -float(clip_val), float(clip_val))
    return lm, IFn


def pass2_process_and_write(
    cfg: dict,
    manifest: List[Dict],
    stats: dict,
    out_root: Path,
    manifests_root: Path,
):
    sr = cfg["sample_rate"]
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    win_length = cfg["win_length"]
    eps = float(cfg["epsilon"])
    write_fp16 = bool(cfg["io"].get("write_fp16", True))

    index = {"train": [], "val": [], "test": []}
    cache_root_cfg = Path(cfg["io"].get("cache_root", "data/cache_npy")).expanduser()
    cache_root = cache_root_cfg / f"sr{sr}"
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_pairs: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    n_bins = n_fft // 2 + 1

    for ex in manifest:
        y, _ = load_audio_resample(ex["path"], sr)
        X = stft_complex(y, n_fft=n_fft, hop=hop, win_length=win_length,
                         window=cfg["window"], center=cfg["center"], pad_mode=cfg["pad_mode"])  # [T,F]
        mag = np.abs(X)
        logmag = compute_logmag(mag, eps=eps)
        phase = np.angle(X)
        phase_u = unwrap_phase_time(phase)
        dphi = phase_u[1:] - phase_u[:-1]
        IF = phase_to_if(dphi)
        logmag = logmag[1:]              # align with IF (T-1,F)
        phase_abs = phase_u[1:]          # absolute phase aligned with logmag/IF
        phase0 = phase_u[0]              # absolute phase of first frame
        T, F = logmag.shape

        logmag_n, IF_n = normalize_features(cfg, stats, logmag, IF)
        if not np.isfinite(logmag_n).all():
            raise ValueError(f"Non-finite logmag after normalization for {ex['id']}")
        if not np.isfinite(IF_n).all():
            raise ValueError(f"Non-finite IF after normalization for {ex['id']}")
        valid = np.ones((T,), dtype=np.uint8)
        logmag_io = logmag_n.astype(np.float16) if write_fp16 else logmag_n.astype(np.float32)
        if_io = IF_n.astype(np.float16) if write_fp16 else IF_n.astype(np.float32)
        phase_io = phase_abs.astype(np.float32)
        phase0_io = phase0.astype(np.float32)

        rel_dir = Path(ex["dataset"]) / ex["split"]
        out_dir = out_root / rel_dir
        out_path = out_dir / f"{ex['id']}.npz"
        _save_npz(
            out_path,
            logmag=logmag_io,
            if_unwrapped=if_io,
            valid_frames=valid,
            phase_abs=phase_io,
            phase0=phase0_io,
            sr=np.int32(sr),
            n_fft=np.int32(n_fft),
            hop=np.int32(hop),
            win_length=np.int32(win_length),
            audio_len=np.int64(len(y)),
            path_orig=np.array(ex["path"]).astype(np.string_),
        )

        cache_base = cache_root / rel_dir / ex["id"]
        cache_base.parent.mkdir(parents=True, exist_ok=True)
        m_path = cache_base.with_suffix(".M.npy")
        i_path = cache_base.with_suffix(".IF.npy")
        p_path = cache_base.with_suffix(".PH.npy")
        p0_path = cache_base.with_suffix(".PH0.npy")
        np.save(m_path, logmag_io)
        np.save(i_path, if_io)
        np.save(p_path, phase_io)
        np.save(p0_path, phase0_io)

        index_entry = {
            "id": ex["id"],
            "path": str(out_path),
            "T": int(T),
            "F": int(F),
            "split": ex["split"],
            "dataset": ex["dataset"],
        }
        index[ex["split"]].append(index_entry)

        manifest_pairs[ex["split"]].append({
            "id": ex["id"],
            "dataset": ex.get("dataset"),
            "split": ex.get("split"),
            "M_path": str(m_path),
            "IF_path": str(i_path),
            "PHI_path": str(p_path),
            "phi0_path": str(p0_path),
            "T": int(T),
            "F": int(F),
            "sr": int(sr),
            "n_fft": int(n_fft),
            "hop": int(hop),
            "win_length": int(win_length),
        })

    # write index files next to processed root
    idx_root = out_root.parent
    idx_dir = idx_root / f"sr{sr}"
    idx_dir.mkdir(parents=True, exist_ok=True)
    for split, items in index.items():
        idx_path = idx_dir / f"{CONFIG['io']['index_prefix']}_{split}.jsonl"
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        with idx_path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    for split, entries in manifest_pairs.items():
        pairs_path = manifests_root / f"{split}_pairs.json"
        payload = {
            "entries": entries,
            "num_files": len(entries),
            "F": entries[0]["F"] if entries else n_bins,
            "sr": sr,
            "n_fft": n_fft,
            "hop": hop,
            "win_length": win_length,
        }
        _save_json(payload, pairs_path)

# =============================
# QA (optional sanity recon)
# =============================

def _istft_reconstruct(cfg: dict, logmag: np.ndarray, IF: np.ndarray) -> np.ndarray:
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    win_length = cfg["win_length"]
    mag = np.power(10.0, logmag.astype(np.float64) / 20.0)
    T, F = mag.shape
    phi = np.zeros((T, F), dtype=np.float64)
    # IF is Δ(unwrapped phase) per hop; cumulative sum (with phi0=0) yields phase
    if T > 1:
        phi[1:] = np.cumsum(IF[1:], axis=0)
    X = mag * np.exp(1j * phi)
    y = librosa.istft(X.T, hop_length=hop, win_length=win_length, window=cfg["window"])
    return y


def run_qa(cfg: dict, out_root: Path, manifests_root: Path, num_examples: int = 4, out_dir: Path = Path("qa")):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_train = manifests_root.parent / f"{cfg['io']['index_prefix']}_train.jsonl"
    if not idx_train.exists():
        print("[QA] train index not found, skipping QA.")
        return
    examples = []
    with idx_train.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            examples.append(json.loads(line))
            if len(examples) >= num_examples:
                break
    report = {"examples": []}
    for ex in examples:
        data = np.load(ex["path"])  # npz
        logmag = data["logmag"].astype(np.float32)
        IF = data["if_unwrapped"].astype(np.float32)
        T = min(len(logmag), len(IF))
        logmag = logmag[:T]
        IF = IF[:T]
        y = _istft_reconstruct(cfg, logmag, IF)
        wav_out = out_dir / f"qa_{ex['id']}.wav"
        sf.write(str(wav_out), y, cfg["sample_rate"])
        report["examples"].append({"id": ex["id"], "npz": ex["path"], "wav": str(wav_out)})
    _save_json(report, out_dir / "qa_report.json")

# =============================
# RUN
# =============================

def run_preprocess(cfg: dict = CONFIG):
    sr = cfg["sample_rate"]
    # 1) Discover
    all_items: List[Dict] = []
    for d in cfg["datasets"]:
        root = Path(d["root"]).expanduser()
        print("[INFO] PASS 1: Discover audio")
        found = discover_audio(root, d.get("pattern", "**/*.wav"), dataset_name=d["name"], min_dur_sec=cfg.get("min_duration_sec", 0.5))
        all_items.extend(found)
    if not all_items:
        print("[IFF-AR][preprocess] No audio found. Check CONFIG['datasets'] paths.")
        return

    # 2) Split & save manifests
    print("[INFO] PASS 2: Split and manifest")
    manifest = make_splits(all_items, cfg["splits"], seed=cfg.get("split_seed", 42))
    manifests_root = Path(cfg["io"]["manifests_root"]) / f"sr{sr}"
    manifests_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        _save_json([ex for ex in manifest if ex["split"] == split], manifests_root / f"{split}.json")

    # 3) Pass1 stats (train only)
    print("[IFF-AR][preprocess] Pass1: collecting stats on train…")
    stats = pass1_collect_stats(cfg, manifest)
    stats_path = manifests_root / f"stats_n{cfg['n_fft']}_h{cfg['hop_length']}.json"
    _save_json(stats, stats_path)

    # 4) Pass2 write processed .npz + indexes
    print("[IFF-AR][preprocess] Pass2: processing and writing .npz…")
    out_root = Path(cfg["io"]["processed_root"]) / f"sr{sr}"
    pass2_process_and_write(cfg, manifest, stats, out_root, manifests_root)

    # 5) QA
    if cfg.get("qa", {}).get("enable", True):
        print("[IFF-AR][preprocess] QA: generating sanity reconstructions…")
        qa_out = Path(cfg["qa"].get("output_dir", "qa"))
        run_qa(cfg, out_root, manifests_root, num_examples=cfg["qa"].get("num_examples", 4), out_dir=qa_out)

    print("[IFF-AR][preprocess] Done.")

# Auto-run on import/execute
run_preprocess(CONFIG)
