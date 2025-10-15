from __future__ import annotations
import os, json, math, time, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable
from collections import OrderedDict

import numpy as np
import torch
import warnings
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

M_ALIASES = [
    "M_norm","logmag_norm","log_mag_norm","log_mag","logmag","M","mag","mag_norm","logM_norm"
]
IF_ALIASES = [
    "IF_norm","if_norm","IF","if","Omega_norm","omega_norm","Omega","omega",
    "dphi_norm","dphi","dphi_dt","phase_deriv","phase_delta"
]
PHI_ALIASES = [
    "phase_abs","phase_unwrapped","phase","phi_abs","phi"
]

def _npz_keys_info(z: np.lib.npyio.NpzFile) -> str:
    lines = []
    for k in z.files:
        try:
            arr = z[k]
            shape = getattr(arr, "shape", None)
            dtype = getattr(arr, "dtype", None)
        except Exception:
            shape, dtype = "?", "?"
        lines.append(f"  - {k}: shape={shape}, dtype={dtype}")
    return "\n".join(lines)

def _pick_arrays_from_npz(npz_path: Path) -> List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    with np.load(npz_path, allow_pickle=False) as z:
        keys = set(z.files)

        def _find_key(aliases: List[str]) -> Optional[str]:
            for a in aliases:
                if a in keys:
                    return a
            pat = re.compile(rf"(^|\W)({'|'.join([re.escape(a) for a in aliases])})(\W|$)", re.IGNORECASE)
            for k in keys:
                if pat.search(k):
                    return k
            return None

        M_key = _find_key(M_ALIASES)
        IF_key = _find_key(IF_ALIASES)

        results: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]] = []

        phi_key = _find_key(PHI_ALIASES)

        if M_key is not None and IF_key is not None:
            M = z[M_key]
            IF = z[IF_key]
            PH = z[phi_key] if phi_key is not None else None
            if M.shape == IF.shape and M.ndim in (2, 3):
                if M.ndim == 2:
                    if PH is not None and PH.shape != M.shape:
                        raise ValueError(f"phase_abs shape mismatch in {npz_path.name}: {PH.shape} vs {M.shape}")
                    results.append(("0", M, IF, PH))
                else:
                    N = M.shape[0]
                    assert IF.shape[0] == N, f"Shape mismatch {M.shape} vs {IF.shape} in {npz_path}"
                    if PH is not None and PH.shape[0] != N:
                        raise ValueError(f"phase_abs shape mismatch in {npz_path.name}: {PH.shape} vs {M.shape}")
                    for n in range(N):
                        PH_n = PH[n] if PH is not None else None
                        results.append((str(n), M[n], IF[n], PH_n))
                return results
            else:
                pass

        for k in z.files:
            A = z[k]
            if A.ndim == 3 and (A.shape[0] == 2 or A.shape[2] == 2):
                if A.shape[0] == 2:
                    M, IF = A[0], A[1]
                    results.append((k, M, IF, None))
                    return results
                if A.shape[2] == 2:
                    M, IF = A[...,0], A[...,1]
                    results.append((k, M, IF, None))
                    return results
            if A.ndim == 4 and (A.shape[1] == 2 or A.shape[3] == 2):
                if A.shape[1] == 2:
                    N = A.shape[0]
                    for n in range(N):
                        M, IF = A[n,0], A[n,1]
                        results.append((f"{k}_{n}", M, IF, None))
                    return results
                if A.shape[3] == 2:
                    N = A.shape[0]
                    for n in range(N):
                        M, IF = A[n,...,0], A[n,...,1]
                        results.append((f"{k}_{n}", M, IF, None))
                    return results

        candidates = [z[k] for k in z.files if getattr(z[k], "ndim", 0) in (2,3)]
        for A in candidates:
            if A.ndim == 2:
                T, F = A.shape
                if F % 2 == 0:
                    M, IF = A[:, :F//2], A[:, F//2:]
                    results.append(("splitF", M, IF, None))
                    return results
                if T % 2 == 0:
                    M, IF = A[:T//2, :], A[T//2:, :]
                    results.append(("splitT", M, IF, None))
                    return results
            if A.ndim == 3:
                continue

        msg = f"\n[build_manifest] Impossibile trovare (M, IF) in: {npz_path.name}\nChiavi disponibili:\n{_npz_keys_info(z)}\n"
        raise KeyError(msg)

def _write_npy_triple(base: Path, suffix: str, M: np.ndarray, IF: np.ndarray, PH: np.ndarray | None) -> Tuple[Path, Path, Path | None, int, int]:
    assert M.shape == IF.shape and M.ndim == 2, f"Expected [T,F], got {M.shape} and {IF.shape}"
    T, F = M.shape
    mpath = base.with_suffix(f".{suffix}.M.npy")
    ipath = base.with_suffix(f".{suffix}.IF.npy")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    np.save(mpath, M)
    np.save(ipath, IF)
    ppath = None
    if PH is not None:
        ppath = base.with_suffix(f".{suffix}.PH.npy")
        np.save(ppath, PH)
    return mpath, ipath, ppath, int(T), int(F)

def build_manifest(
    data_root: str,
    cache_root: str,
    manifest_path: str,
    accept_npy_pairs: bool = True,
) -> Dict[str, Any]:
    """
    Scansiona data_root:
      - se trova .npz, estrae coppie (M,IF) con heuristiche robuste
      - crea entry multiple per .npz multi-clip
      - altrimenti, se trova .npy coppie *.M.npy / *.IF.npy, le usa direttamente
    Scrive un manifest JSON con (id, M_path, IF_path, T, F).
    """
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []

    npz_files = sorted(data_root.rglob("*.npz"))
    for p in npz_files:
        base = cache_root / p.stem
        try:
            pairs = _pick_arrays_from_npz(p)
        except Exception as e:
            print(str(e))
            continue
        for suffix, M, IF, PH in pairs:
            if M.ndim != 2:
                print(f"[build_manifest][warn] Skipping {p.name}:{suffix} shape={M.shape} (atteso [T,F])")
                continue
            mpath, ipath, ppath, T, F = _write_npy_triple(base, suffix, M, IF, PH)
            entries.append({
                "id": f"{p.stem}_{suffix}",
                "M_path": str(mpath),
                "IF_path": str(ipath),
                "PHI_path": str(ppath) if ppath is not None else None,
                "phi0_path": None,
                "T": T, "F": F
            })

    if accept_npy_pairs:
        npy_M_files = sorted(data_root.rglob("*.M.npy"))
        for M_path in npy_M_files:
            IF_path = Path(str(M_path).replace(".M.npy", ".IF.npy"))
            if not IF_path.exists():
                continue
            PH_path = Path(str(M_path).replace(".M.npy", ".PH.npy"))
            if not PH_path.exists():
                PH_path = None
            PH0_path = Path(str(M_path).replace(".M.npy", ".PH0.npy"))
            if not PH0_path.exists():
                PH0_path = None
            try:
                M = np.load(M_path, allow_pickle=False)
                T, F = M.shape
            except Exception as e:
                print(f"[build_manifest][warn] Skipping {M_path}: {e}")
                continue
            entries.append({
                "id": Path(M_path).stem.replace(".M", ""),
                "M_path": str(M_path),
                "IF_path": str(IF_path),
                "PHI_path": str(PH_path) if PH_path is not None else None,
                "phi0_path": str(PH0_path) if PH0_path is not None else None,
                "T": int(T), "F": int(F)
            })

    if not entries:
        raise FileNotFoundError(
            "Nessuna entry valida trovata. Verifica i .npz: devono contenere (logmag, IF) "
            "oppure fornisci coppie .M.npy/.IF.npy. Vedi messaggi sopra per le chiavi disponibili."
        )

    manifest = {
        "entries": entries,
        "num_files": len(entries),
        "F": entries[0]["F"],
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_manifest] OK: {len(entries)} entries")
    return manifest


_MEMMAP_FALLBACK_WARNINGS: set[str] = set()
_CACHE_DEFAULT_LIMIT = int(os.environ.get("IFFAR_DATASET_CACHE_LIMIT", "32"))


def _close_memmap(arr: np.ndarray) -> None:
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        try:
            mm.close()
        except Exception:
            pass


def _load_array(path: str) -> np.ndarray:
    for mode in ("r+", "r"):
        try:
            return np.load(path, mmap_mode=mode, allow_pickle=False)
        except Exception:
            continue
    arr = np.load(path, allow_pickle=False)
    if path not in _MEMMAP_FALLBACK_WARNINGS:
        warnings.warn(f"[dataloader] mmap unavailable for {path}; loaded into RAM", RuntimeWarning)
        _MEMMAP_FALLBACK_WARNINGS.add(path)
    return arr


class _ArrayCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str) -> np.ndarray:
        arr = self._store.pop(path, None)
        if arr is not None:
            self._store[path] = arr
            return arr
        arr = _load_array(path)
        if self.max_items <= 0:
            return arr
        self._store[path] = arr
        if len(self._store) > self.max_items:
            _, old = self._store.popitem(last=False)
            _close_memmap(old)
        return arr


class PackedWindowsDataset(Dataset):
    """
    Restituisce finestre causali con K target consecutivi per ridurre overhead.
      - M_ctx:  [L, F] (fp16)
      - IF_ctx: [L, F] (fp16)
      - M_tgt:  [K, F] (fp16)
      - IF_tgt: [K, F] (fp16)
    """
    def __init__(
        self,
        manifest_path: str,
        L: int,
        K: int,
        stride: int = 1,
        max_items_per_file: Optional[int] = None,
        max_total_items: Optional[int] = None,
        file_shuffle: bool = True,
        use_half: bool = True,
        seed: int = 1234,
    ):
        super().__init__()
        with open(manifest_path, "r") as f:
            mani = json.load(f)
        if "entries" not in mani or not mani["entries"]:
            raise ValueError(f"Manifest vuoto o invalido: {manifest_path}")
        self.entries = mani["entries"]
        if file_shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.entries)

        self.L = int(L)
        self.K = int(K)
        self.stride = int(stride)
        self.max_items_per_file = max_items_per_file
        self.max_total_items = int(max_total_items) if max_total_items is not None else None
        if self.max_total_items is not None and self.max_total_items <= 0:
            self.max_total_items = None
        self.use_half = use_half
        self._cache_limit = _CACHE_DEFAULT_LIMIT

        self.index: List[Tuple[int, int]] = []
        for i, e in enumerate(self.entries):
            T = int(e["T"])
            need = self.L + self.K
            npos = (T - need) // self.stride + 1
            if npos <= 0:
                continue
            if max_items_per_file is not None:
                npos = min(npos, max_items_per_file)
            starts = np.arange(0, npos * self.stride, self.stride, dtype=np.int64)
            if self.max_total_items is not None:
                remaining = self.max_total_items - len(self.index)
                if remaining <= 0:
                    break
                starts = starts[:remaining]
            self.index.extend([(i, int(s)) for s in starts])
            if self.max_total_items is not None and len(self.index) >= self.max_total_items:
                break

        self._worker_local_cache: Optional[Dict[str, _ArrayCache]] = None

    def __len__(self) -> int:
        return len(self.index)

    def _ensure_worker_cache(self) -> None:
        if self._worker_local_cache is None:
            limit = self._cache_limit
            self._worker_local_cache = {
                "M": _ArrayCache(limit),
                "IF": _ArrayCache(limit),
                "PHI": _ArrayCache(limit),
            }

    def _get_arrays(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_worker_cache()
        e = self.entries[file_idx]
        Mp = e["M_path"]; IFp = e["IF_path"]; PHp = e.get("PHI_path")
        if PHp is None:
            raise KeyError(f"PHI_path missing for entry {e['id']}")
        M_cache = self._worker_local_cache["M"]
        IF_cache = self._worker_local_cache["IF"]
        PH_cache = self._worker_local_cache["PHI"]
        M_arr = M_cache.get(Mp)
        IF_arr = IF_cache.get(IFp)
        PH_arr = PH_cache.get(PHp)
        return M_arr, IF_arr, PH_arr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, t0 = self.index[idx]
        Mmm, IFmm, PHmm = self._get_arrays(file_idx)
        L, K = self.L, self.K

        sl = slice(t0, t0 + L + K)
        M_slice  = Mmm[sl]
        IF_slice = IFmm[sl]
        PH_slice = PHmm[sl]

        if not np.isfinite(M_slice).all() or not np.isfinite(IF_slice).all() or not np.isfinite(PH_slice).all():
            raise ValueError(f"Non-finite values detected in dataset entry {self.entries[file_idx]['id']} at slice starting {t0}")

        M_ctx, IF_ctx = M_slice[:L],  IF_slice[:L]
        M_tgt, IF_tgt = M_slice[L:],  IF_slice[L:]
        PH_ctx, PH_tgt = PH_slice[:L], PH_slice[L:]

        M_ctx_t  = torch.from_numpy(M_ctx)
        IF_ctx_t = torch.from_numpy(IF_ctx)
        M_tgt_t  = torch.from_numpy(M_tgt)
        IF_tgt_t = torch.from_numpy(IF_tgt)
        PH_ctx_t = torch.from_numpy(PH_ctx).float()
        PH_tgt_t = torch.from_numpy(PH_tgt).float()

        if self.use_half:
            M_ctx_t  = M_ctx_t.to(torch.float16, copy=False)
            IF_ctx_t = IF_ctx_t.to(torch.float16, copy=False)
            M_tgt_t  = M_tgt_t.to(torch.float16, copy=False)
            IF_tgt_t = IF_tgt_t.to(torch.float16, copy=False)

        return {
            "M_ctx":  M_ctx_t,
            "IF_ctx": IF_ctx_t,
            "M_tgt":  M_tgt_t,
            "IF_tgt": IF_tgt_t,
            "phi_ctx": PH_ctx_t,
            "phi_tgt": PH_tgt_t,
            "frame_start_tgt": torch.tensor(int(t0 + self.L), dtype=torch.int32),
            "entry_id": self.entries[file_idx]["id"],
        }

def packed_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        values = [b[k] for b in batch]
        if isinstance(values[0], torch.Tensor):
            out[k] = torch.stack(values, dim=0)
        else:
            out[k] = values
    return out

def _seed_worker(worker_id: int, base_seed: int | None = None):
    worker_info = get_worker_info()
    seed = torch.initial_seed() % 2**32 if base_seed is None else base_seed % 2**32
    worker_offset = worker_info.id if worker_info is not None else worker_id
    np.random.seed((seed + worker_offset) % 2**32)

def make_dataloader(
    manifest_path: str,
    L: int,
    K: int,
    batch_size: int = 2,
    stride: int = 1,
    max_items_per_file: Optional[int] = None,
    max_total_items: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: bool = False,
    prefetch_factor: int = 6,
    persistent_workers: bool = True,
    file_shuffle: bool = True,
    use_half: bool = True,
    worker_seed: int = 1234,
    sampler: Optional[torch.utils.data.Sampler] = None,
    drop_last: bool = True,
) -> Tuple[DataLoader, Dataset]:
    if num_workers is None:
        cpu = os.cpu_count() or 8
        num_workers = 8

    ds = PackedWindowsDataset(
        manifest_path=manifest_path,
        L=L, K=K,
        stride=stride,
        max_items_per_file=max_items_per_file,
        max_total_items=max_total_items,
        file_shuffle=file_shuffle,
        use_half=use_half,
        seed=worker_seed,
    )
    
    loader_args = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=packed_collate,
        drop_last=drop_last,
    )
    if sampler is not None:
        loader_args["sampler"] = sampler

    if num_workers > 0:
        if prefetch_factor is not None:
            loader_args["prefetch_factor"] = prefetch_factor
        loader_args["persistent_workers"] = persistent_workers
        if worker_seed is not None:
            loader_args["worker_init_fn"] = lambda wid: _seed_worker(wid, worker_seed)

    loader = DataLoader(**loader_args)
    return loader, ds


class ShardedIterable(IterableDataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        L: int,
        K: int,
        stride: int = 1,
        use_half: bool = True,
    ):
        super().__init__()
        self.pairs = pairs
        self.L = int(L)
        self.K = int(K)
        self.stride = int(stride)
        self.use_half = use_half

    def _yield_file(self, Mmm: np.memmap, IFmm: np.memmap) -> Iterable[Dict[str, torch.Tensor]]:
        T, F = Mmm.shape
        need = self.L + self.K
        if T < need:
            return
        info = get_worker_info()
        if info is None or info.num_workers <= 1:
            starts = range(0, T - need + 1, self.stride)
        else:
            starts = [s for i, s in enumerate(range(0, T - need + 1, self.stride)) if (i % info.num_workers) == info.id]

        for t0 in starts:
            sl = slice(t0, t0 + need)
            M_slice  = Mmm[sl]
            IF_slice = IFmm[sl]
            M_ctx, IF_ctx = M_slice[:self.L],  IF_slice[:self.L]
            M_tgt, IF_tgt = M_slice[self.L:],  IF_slice[self.L:]

            M_ctx_t  = torch.from_numpy(M_ctx)
            IF_ctx_t = torch.from_numpy(IF_ctx)
            M_tgt_t  = torch.from_numpy(M_tgt)
            IF_tgt_t = torch.from_numpy(IF_tgt)

            if self.use_half:
                M_ctx_t  = M_ctx_t.to(torch.float16, copy=False)
                IF_ctx_t = IF_ctx_t.to(torch.float16, copy=False)
                M_tgt_t  = M_tgt_t.to(torch.float16, copy=False)
                IF_tgt_t = IF_tgt_t.to(torch.float16, copy=False)

            yield {
                "M_ctx":  M_ctx_t,
                "IF_ctx": IF_ctx_t,
                "M_tgt":  M_tgt_t,
                "IF_tgt": IF_tgt_t,
            }

    def __iter__(self):
        for M_path, IF_path in self.pairs:
            Mmm  = np.load(M_path, allow_pickle=False)
            IFmm = np.load(IF_path, allow_pickle=False)
            yield from self._yield_file(Mmm, IFmm)

def make_iter_loader_from_manifest(
    manifest_path: str,
    L: int,
    K: int,
    batch_size: int = 2,
    stride: int = 1,
    num_workers: Optional[int] = None,
    prefetch_factor: int = 6,
    persistent_workers: bool = True,
    use_half: bool = True,
) -> Tuple[DataLoader, IterableDataset]:
    with open(manifest_path, "r") as f:
        mani = json.load(f)
    if "entries" not in mani or not mani["entries"]:
        raise ValueError(f"Manifest vuoto o invalido: {manifest_path}")
    pairs = [(e["M_path"], e["IF_path"]) for e in mani["entries"]]
    ds = ShardedIterable(pairs, L=L, K=K, stride=stride, use_half=use_half)
    if num_workers is None:
        cpu = os.cpu_count() or 8
        num_workers = max(2, min(8, cpu - 1))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=False,
        drop_last=True,
    )
    return loader, ds


def _mini_bench(loader: DataLoader, steps: int = 64) -> float:
    t0 = time.time()
    it = 0
    for i, batch in enumerate(loader):
        _ = batch["M_ctx"].shape
        it += 1
        if it >= steps:
            break
    t1 = time.time()
    return it / max(1e-6, (t1 - t0))


__all__ = [
    "build_manifest",
    "PackedWindowsDataset",
    "packed_collate",
    "make_dataloader",
    "ShardedIterable",
    "make_iter_loader_from_manifest",
]

# Quick test
# if __name__ == "__main__":

#     try:
#         mani = build_manifest(
#             data_root="data/processed/sr16000/nsynth/train",
#             cache_root="data/cache_npy",
#             manifest_path="manifests/sr16000/stats_n1024_h256.json",
#         )
#         print(f"[manifest] files: {mani['num_files']}, F={mani['F']}")
#     except Exception as e:
#         print(f"[manifest] ERROR: {e}")

#     L, K = 32, 12
#     try:
#         loader, ds = make_dataloader(
#             manifest_path="manifests/sr16000/stats_n1024_h256.json",
#             L=L, K=K, batch_size=2, stride=1,
#             num_workers=max(2, (os.cpu_count() or 8) - 1),
#             prefetch_factor=6,
#             persistent_workers=True,
#         )
#         thr = _mini_bench(loader, steps=64)
#         print(f"[bench] PackedWindows loader: {thr:.2f} it/s")
#     except Exception as e:
#         print(f"[bench] loader ERROR: {e}")

#     try:
#         iter_loader, _ = make_iter_loader_from_manifest(
#             manifest_path="manifests/sr16000/stats_n1024_h256.json",
#             L=L, K=K, batch_size=2, stride=1,
#         )
#         thr2 = _mini_bench(iter_loader, steps=64)
#         print(f"[bench] Iterable loader: {thr2:.2f} it/s")
#     except Exception as e:
#         print(f"[bench] iter ERROR: {e}")