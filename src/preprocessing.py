from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm

CONFIG = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "window": "hann",
    "center": True,
    "pad_mode": "reflect",
    "epsilon": 1.0e-7,

    "datasets": [
        {"name": "nsynth", "root": "dataset/nsynth-train/audio", "pattern": "**/*.wav"},
    ],

    "splits": {"train": 0.9, "val": 0.05, "test": 0.05},
    "split_seed": 42,
    "min_duration_sec": 0.5,

    "logmag_norm": {"type": "global", "clip_quantiles": [0.01, 0.99]},
    "if_norm": {"type": "per_bin", "estimator": "mad", "clip_quantiles": [0.01, 0.99], "clip_value": 50.0},

    "io": {
        "processed_root": "data/processed",
        "manifests_root": "manifests",
        "cache_root": "data/cache_npy",
        "index_prefix": "index",
        "write_fp16": True,
    },

    "qa": {"enable": False, "num_examples": 4, "output_dir": "qa"},
}

# Utility
def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    np.savez_compressed(path, **arrays)

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


# Parallel helpers
def _torch_set_threads(num_threads: int = 1) -> None:
    """Force torch and common math libs to single-thread mode within workers."""
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = str(num_threads)
    torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(num_threads)


def get_hann_window(win_length: int, window: str = "hann") -> torch.Tensor:
    """Return a Hann window tensor on CPU for reuse across workers."""
    if window.lower() != "hann":
        raise ValueError(f"Only Hann window is supported in HPC path, got: {window}")
    return torch.hann_window(win_length, periodic=True, dtype=torch.float32)


# STFT & feature extraction
def stft_complex(y: np.ndarray, n_fft: int, hop: int, win_length: int, window: str = "hann", center: bool = True, pad_mode: str = "reflect") -> np.ndarray:
    X = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    return X.T


def compute_logmag(mag: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Log-magnitude in dB for numpy arrays."""
    return 20.0 * np.log10(np.clip(mag, a_min=eps, a_max=None))


def compute_logmag_torch(mag: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return 20.0 * torch.log10(torch.clamp(mag, min=eps))

def unwrap_phase_time(phase_tf: np.ndarray) -> np.ndarray:
    return np.unwrap(phase_tf, axis=0)


def phase_to_if(dphi_tf: np.ndarray) -> np.ndarray:
    """Instantaneous frequency as unwrapped phase derivative per hop (rad / hop)."""
    return dphi_tf.astype(np.float32)

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


# Pass1 parallel workers -------------------------------------------------------
_PASS1_CTX: Optional[Dict[str, object]] = None


def _pass1_worker_initializer(ctx: Dict[str, object]) -> None:
    """Prepare global context for pass1 workers."""
    _torch_set_threads(1)
    ctx_local = dict(ctx)
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
    except ImportError:
        ctx_local["threadpool_limits"] = None
    else:
        ctx_local["threadpool_limits"] = threadpool_limits
    ctx_local["window_tensor"] = get_hann_window(ctx_local["win_length"], ctx_local["window"])
    global _PASS1_CTX
    _PASS1_CTX = ctx_local


def _pass1_worker_core(entry: Dict[str, object], ctx: Dict[str, object]) -> Dict[str, object]:
    """Compute downsampled logmag and IF samples for statistics."""
    y, sr = sf.read(entry["path"], dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != ctx["sr"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=ctx["sr"], res_type="kaiser_best")
        sr = ctx["sr"]
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = (y / peak).astype(np.float32)

    waveform = torch.from_numpy(y)
    stft = torch.stft(
        waveform,
        n_fft=ctx["n_fft"],
        hop_length=ctx["hop"],
        win_length=ctx["win_length"],
        window=ctx["window_tensor"],
        center=ctx["center"],
        pad_mode=ctx["pad_mode"],
        return_complex=True,
    )
    X = stft.transpose(0, 1).cpu().numpy()
    mag = np.abs(X)
    logmag = compute_logmag(mag, eps=ctx["eps"])
    phase = np.angle(X)
    phase_u = unwrap_phase_time(phase)
    if phase_u.shape[0] < 2:
        raise ValueError(f"STFT returned less than two frames for {entry['path']}")
    dphi = phase_u[1:] - phase_u[:-1]
    IF = phase_to_if(dphi)
    logmag = logmag[1:]  # align to IF frames
    T = logmag.shape[0]
    if T == 0:
        raise ValueError(f"No STFT frames after alignment for {entry['path']}")

    max_frames = ctx["max_frames"]
    if T <= max_frames:
        idx = slice(None)
    else:
        idx = np.linspace(0, T - 1, num=max_frames, dtype=np.int64)
    logmag_sample = np.ascontiguousarray(logmag[idx], dtype=np.float32)
    if_sample = np.ascontiguousarray(IF[idx], dtype=np.float32)

    return {
        "ok": True,
        "path": entry["path"],
        "logmag": logmag_sample,
        "if": if_sample,
        "frames": int(T),
    }


def _pass1_worker(entry: Dict[str, object]) -> Dict[str, object]:
    """Wrapper to ensure threadpool limiting exceptions are handled cleanly."""
    ctx = _PASS1_CTX
    if ctx is None:
        raise RuntimeError("Pass1 worker context not initialized.")
    limiter = ctx.get("threadpool_limits")
    try:
        if limiter:
            with limiter(limits=1):
                return _pass1_worker_core(entry, ctx)
        return _pass1_worker_core(entry, ctx)
    except Exception as exc:
        return {"ok": False, "path": entry["path"], "err": str(exc)}

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

def pass1_collect_stats_parallel(cfg: dict, manifest: List[Dict], max_workers: int = 32) -> dict:
    """
    Parallel, HPC-ready statistics collection on training split only.
    Questa versione non accumula tutti i campioni in RAM, ma aggiorna
    le statistiche in streaming (merge online). Evita OOM su dataset grandi.
    """

    # --- Setup ----------------------------------------------------------------
    sr = cfg["sample_rate"]
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    win_length = cfg["win_length"]
    eps = float(cfg["epsilon"])

    random.seed(cfg.get("split_seed", 42))
    np.random.seed(cfg.get("split_seed", 42))

    # Filtra solo i file del train
    train_items = [ex for ex in manifest if ex.get("split") == "train"]
    if not train_items:
        raise RuntimeError("No training items available for statistics collection.")
    train_items = sorted(train_items, key=lambda ex: (ex["dataset"], ex["id"]))

    # Context condiviso tra i worker
    ctx_payload = {
        "sr": sr,
        "n_fft": n_fft,
        "hop": hop,
        "win_length": win_length,
        "window": cfg["window"],
        "center": cfg["center"],
        "pad_mode": cfg["pad_mode"],
        "eps": eps,
        "max_frames": 512,
    }

    # Collector streaming
    rs = RunningStats(
        clip_quantiles=tuple(cfg["logmag_norm"].get("clip_quantiles", []))
        if cfg["logmag_norm"].get("clip_quantiles")
        else None
    )
    if_cfg = cfg["if_norm"]
    rb = RobustPerBin(
        n_bins=n_fft // 2 + 1,
        estimator=if_cfg.get("estimator", "mad"),
        clip_quantiles=tuple(if_cfg.get("clip_quantiles", []))
        if if_cfg.get("clip_quantiles")
        else None,
    )

    failures: List[Tuple[str, str]] = []

    # --- Parallel pool ---------------------------------------------------------
    mp_ctx = get_context("fork")
    total = len(train_items)
    cpu_cap = os.cpu_count() or max_workers
    max_workers = min(max_workers, cpu_cap, total)
    chunk_size = max(1, total // (max_workers * 4)) if total >= max_workers else 1

    print(f"[Pass1] Starting streaming stats collection with {max_workers} workers...")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
        initializer=_pass1_worker_initializer,
        initargs=(ctx_payload,),
    ) as executor:
        results = executor.map(_pass1_worker, train_items, chunksize=chunk_size)
        for res in tqdm(results, total=total, desc="[Pass1] stats", dynamic_ncols=True):
            if not res["ok"]:
                failures.append((res["path"], res["err"]))
                continue
            rs.update(res["logmag"])
            rb.update(res["if"])

    # --- Finalizzazione e merge -------------------------------------------------
    rs.finalize()
    stats = {
        "logmag": {
            "type": "global",
            "mean": float(rs.mean),
            "std": float(rs.std),
            "clip_quantiles": cfg["logmag_norm"].get("clip_quantiles"),
        },
        "if_unwrapped": rb.to_dict(),
        "meta": {"sr": sr, "n_fft": n_fft, "hop": hop, "win_length": win_length},
    }

    if failures:
        print(f"[Pass1] {len(failures)} files failed during processing:")
        for path, err in failures[:5]:
            print(f"   - {path}: {err}")

    print("[Pass1] Streaming stats collection completed successfully.")
    return stats



def pass1_collect_stats(cfg: dict, manifest: List[Dict]) -> dict:
    """Backward-compatible alias."""
    return pass1_collect_stats_parallel(cfg, manifest)

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

# Pass2 parallel workers ------------------------------------------------------
_PASS2_CTX: Optional[Dict[str, object]] = None


def _pass2_worker_initializer(ctx: Dict[str, object]) -> None:
    """Prepare global context for pass2 workers."""
    _torch_set_threads(1)
    ctx_local = dict(ctx)
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
    except ImportError:
        ctx_local["threadpool_limits"] = None
    else:
        ctx_local["threadpool_limits"] = threadpool_limits
    ctx_local["window_tensor"] = get_hann_window(ctx_local["win_length"], ctx_local["window"])
    ctx_local["out_root"] = Path(ctx_local["out_root"])
    ctx_local["cache_root"] = Path(ctx_local["cache_root"])
    global _PASS2_CTX
    _PASS2_CTX = ctx_local


def _pass2_worker_core(entry: Dict[str, object], ctx: Dict[str, object]) -> Dict[str, object]:
    """Full feature extraction, normalization, and disk writes for one item."""
    y, sr = sf.read(entry["path"], dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != ctx["sr"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=ctx["sr"], res_type="kaiser_best")
        sr = ctx["sr"]
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = (y / peak).astype(np.float32)

    waveform = torch.from_numpy(y)
    stft = torch.stft(
        waveform,
        n_fft=ctx["n_fft"],
        hop_length=ctx["hop"],
        win_length=ctx["win_length"],
        window=ctx["window_tensor"],
        center=ctx["center"],
        pad_mode=ctx["pad_mode"],
        return_complex=True,
    )
    X = stft.transpose(0, 1).cpu().numpy()
    mag = np.abs(X)
    logmag = compute_logmag(mag, eps=ctx["eps"])
    phase = np.angle(X)
    phase_u = unwrap_phase_time(phase)
    if phase_u.shape[0] < 2:
        raise ValueError(f"STFT returned less than two frames for {entry['path']}")
    dphi = phase_u[1:] - phase_u[:-1]
    IF = phase_to_if(dphi)
    logmag = logmag[1:]
    phase_abs = phase_u[1:]
    phase0 = phase_u[0]
    T, F = logmag.shape
    if T == 0:
        raise ValueError(f"No STFT frames after alignment for {entry['path']}")

    logmag_n, IF_n = normalize_features(ctx["cfg"], ctx["stats"], logmag, IF)
    if not np.isfinite(logmag_n).all():
        raise ValueError(f"Non-finite logmag after normalization for {entry['path']}")
    if not np.isfinite(IF_n).all():
        raise ValueError(f"Non-finite IF after normalization for {entry['path']}")

    valid = np.ones((T,), dtype=np.uint8)
    if ctx["write_fp16"]:
        logmag_io = logmag_n.astype(np.float16)
        if_io = IF_n.astype(np.float16)
    else:
        logmag_io = logmag_n.astype(np.float32)
        if_io = IF_n.astype(np.float32)
    phase_io = phase_abs.astype(np.float32)
    phase0_io = phase0.astype(np.float32)

    rel_dir = Path(entry["dataset"]) / entry["split"]
    out_dir = ctx["out_root"] / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{entry['id']}.npz"
    _save_npz(
        out_path,
        logmag=logmag_io,
        if_unwrapped=if_io,
        valid_frames=valid,
        phase_abs=phase_io,
        phase0=phase0_io,
        sr=np.int32(ctx["sr"]),
        n_fft=np.int32(ctx["n_fft"]),
        hop=np.int32(ctx["hop"]),
        win_length=np.int32(ctx["win_length"]),
        audio_len=np.int64(len(y)),
        path_orig=np.array(entry["path"]).astype(np.string_),
    )

    cache_base = ctx["cache_root"] / rel_dir / entry["id"]
    cache_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_base.with_suffix(".M.npy"), logmag_io)
    np.save(cache_base.with_suffix(".IF.npy"), if_io)
    np.save(cache_base.with_suffix(".PH.npy"), phase_io)
    np.save(cache_base.with_suffix(".PH0.npy"), phase0_io)

    index_entry = {
        "id": entry["id"],
        "path": str(out_path),
        "T": int(T),
        "F": int(F),
        "split": entry["split"],
        "dataset": entry["dataset"],
    }
    pair_entry = {
        "id": entry["id"],
        "dataset": entry["dataset"],
        "split": entry["split"],
        "M_path": str(cache_base.with_suffix(".M.npy")),
        "IF_path": str(cache_base.with_suffix(".IF.npy")),
        "PHI_path": str(cache_base.with_suffix(".PH.npy")),
        "phi0_path": str(cache_base.with_suffix(".PH0.npy")),
        "T": int(T),
        "F": int(F),
        "sr": int(ctx["sr"]),
        "n_fft": int(ctx["n_fft"]),
        "hop": int(ctx["hop"]),
        "win_length": int(ctx["win_length"]),
    }
    return {"ok": True, "split": entry["split"], "index": index_entry, "pair": pair_entry}


def _pass2_worker(entry: Dict[str, object]) -> Dict[str, object]:
    """Wrapper to guard against worker-side failures."""
    ctx = _PASS2_CTX
    if ctx is None:
        raise RuntimeError("Pass2 worker context not initialized.")
    limiter = ctx.get("threadpool_limits")
    try:
        if limiter:
            with limiter(limits=1):
                return _pass2_worker_core(entry, ctx)
        return _pass2_worker_core(entry, ctx)
    except Exception as exc:
        return {"ok": False, "path": entry["path"], "err": str(exc)}


def pass2_process_and_write(
    cfg: dict,
    manifest: List[Dict],
    stats: dict,
    out_root: Path,
    manifests_root: Path,
):
    """
    Parallel feature extraction and serialization across the full manifest.
    """
    # --- Setup ----------------------------------------------------------------
    sr = cfg["sample_rate"]
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    win_length = cfg["win_length"]
    eps = float(cfg["epsilon"])
    write_fp16 = bool(cfg["io"].get("write_fp16", True))

    out_root.mkdir(parents=True, exist_ok=True)
    cache_root_cfg = Path(cfg["io"].get("cache_root", "data/cache_npy")).expanduser()
    cache_root = cache_root_cfg / f"sr{sr}"
    cache_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    index: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    manifest_pairs: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    n_bins = n_fft // 2 + 1

    total = len(manifest)
    if total == 0:
        return

    ctx_payload = {
        "cfg": cfg,
        "stats": stats,
        "sr": sr,
        "n_fft": n_fft,
        "hop": hop,
        "win_length": win_length,
        "window": cfg["window"],
        "center": cfg["center"],
        "pad_mode": cfg["pad_mode"],
        "eps": eps,
        "write_fp16": write_fp16,
        "out_root": str(out_root),
        "cache_root": str(cache_root),
    }

    payloads = [
        {"id": ex["id"], "path": ex["path"], "dataset": ex["dataset"], "split": ex["split"]}
        for ex in manifest
    ]

    # --- Pool -----------------------------------------------------------------
    mp_start = "spawn" if sys.platform.startswith("darwin") else "fork"
    mp_ctx = get_context(mp_start)
    cpu_cap = os.cpu_count() or 32
    pool_workers = min(32, cpu_cap, total)
    chunk_size = max(1, total // (pool_workers * 4)) if total >= pool_workers else 1
    failures: List[Tuple[str, str]] = []

    with ProcessPoolExecutor(
        max_workers=pool_workers,
        mp_context=mp_ctx,
        initializer=_pass2_worker_initializer,
        initargs=(ctx_payload,),
    ) as executor:
        results = executor.map(_pass2_worker, payloads, chunksize=chunk_size)
        for res in tqdm(results, total=total, desc="[Pass2] processing", dynamic_ncols=True):
            if not res["ok"]:
                failures.append((res["path"], res["err"]))
                continue
            split = res["split"]
            index[split].append(res["index"])
            manifest_pairs[split].append(res["pair"])

    if failures:
        for path, err in failures:
            print(f"[Pass2][warn] Failed to process {path}: {err}")

    # --- Manifest write-out ---------------------------------------------------
    idx_root = out_root.parent
    idx_dir = idx_root / f"sr{sr}"
    idx_dir.mkdir(parents=True, exist_ok=True)
    index_prefix = cfg["io"].get("index_prefix", "index")
    for split, items in index.items():
        idx_path = idx_dir / f"{index_prefix}_{split}.jsonl"
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

def _istft_reconstruct(cfg: dict, logmag: np.ndarray, IF: np.ndarray) -> np.ndarray:
    n_fft = cfg["n_fft"]
    hop = cfg["hop_length"]
    win_length = cfg["win_length"]
    mag = np.power(10.0, logmag.astype(np.float64) / 20.0)
    T, F = mag.shape
    phi = np.zeros((T, F), dtype=np.float64)
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

def run_preprocess(cfg: dict = CONFIG):
    sr = cfg["sample_rate"]
    all_items: List[Dict] = []
    for d in cfg["datasets"]:
        root = Path(d["root"]).expanduser()
        found = discover_audio(root, d.get("pattern", "**/*.wav"), dataset_name=d["name"], min_dur_sec=cfg.get("min_duration_sec", 0.5))
        all_items.extend(found)
    if not all_items:
        print("[IFF-AR][preprocess] No audio found. Check CONFIG['datasets'] paths.")
        return

    manifest = make_splits(all_items, cfg["splits"], seed=cfg.get("split_seed", 42))
    manifests_root = Path(cfg["io"]["manifests_root"]) / f"sr{sr}"
    manifests_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        _save_json([ex for ex in manifest if ex["split"] == split], manifests_root / f"{split}.json")

    print("[IFF-AR][preprocess] Pass1: collecting stats on train…")
    stats = pass1_collect_stats_parallel(cfg, manifest)
    stats_path = manifests_root / f"stats_n{cfg['n_fft']}_h{cfg['hop_length']}.json"
    _save_json(stats, stats_path)

    print("[IFF-AR][preprocess] Pass2: processing and writing .npz…")
    out_root = Path(cfg["io"]["processed_root"]) / f"sr{sr}"
    pass2_process_and_write(cfg, manifest, stats, out_root, manifests_root)

    if cfg.get("qa", {}).get("enable", True):
        print("[IFF-AR][preprocess] QA: generating sanity reconstructions…")
        qa_out = Path(cfg["qa"].get("output_dir", "qa"))
        run_qa(cfg, out_root, manifests_root, num_examples=cfg["qa"].get("num_examples", 4), out_dir=qa_out)

    print("[IFF-AR][preprocess] Done.")

run_preprocess(CONFIG)
