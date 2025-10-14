from __future__ import annotations
import os, json, gc, time
from pathlib import Path
from typing import Dict, Optional
from contextlib import nullcontext
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from collections import defaultdict

from dataloader import PackedWindowsDataset, packed_collate, _seed_worker
from model.iffar import IFFARModel
from model.loss import IFFARLoss
from model.phase_rec import PhaseReconstructor
from train_utils import setup_distributed, cleanup_distributed, log_rank0, barrier, is_rank0

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

CONFIG: Dict = {
    "paths": {
        "train_data_root": "/leonardo_work/try25_santini/Deep-Learning/data/processed/sr16000/nsynth/train",
        "val_data_root":   "/leonardo_work/try25_santini/Deep-Learning/data/processed/sr16000/nsynth/val",
        "manifest_train":  "/leonardo_work/try25_santini/Deep-Learning/manifests/sr16000/train_pairs.json",
        "manifest_val":    "/leonardo_work/try25_santini/Deep-Learning/manifests/sr16000/val_pairs.json",
        "stats_path":      "/leonardo_work/try25_santini/Deep-Learning/manifests/sr16000/stats_n2048_h512.json",
        "ckpt_dir":        "/leonardo_work/try25_santini/Deep-Learning/checkpoints",
        "resume_ckpt":     "",
    },

    "data": { 
        "L": 24,
        "K": 4,
        "batch_size": 32,
        "stride": 128,
        "num_workers": 2,
        "max_train_items": 10000,
        "max_val_items": 10000,
    },

    "model": {
        "d_model": 512,
        "n_layers": 12,
        "kernel_size": 7,
        "dropout": 0.05,
        "n_film_bands": 32,
        
        "flow_layers": 8,
        "flow_hidden": 256,
        "flow_kernel_size": 5,
        "flow_use_tanh_scale": True,
        "flow_scale_factor": 1.0,
        
        "mag_hidden": 256,
        "mag_use_prev_skip": True,
        "mag_predict_logvar": True,
        "mag_spectral_smoothing": True,
        "mag_kernel": 7,
        "mag_dropout": 0.05,
        
        "n_fft": 2048,
        "hop": 512,
        "win_length": 2048,
        "center": True,
    },

    "optim": {
        "epochs": 10,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "betas": (0.9, 0.98),
        "grad_accum": 4,
        "max_grad_norm": 1.0,
        "warmup_steps": 1000,
    },

    "loss": {
        "lambda_if": 1.0,
        "lambda_if_reg": 0.25,   
        "lambda_mag": 10.0,
        "lambda_cons": 0.75,
        "lambda_overlap": 0.5,
        "lambda_time": 0.05,
        "mag_loss": "huber",
        "mag_huber_delta": 1.0,
        "time_alpha_sisdr": 0.5,
        "time_beta_l1": 0.5,
        "cons_mode": "both",
        "cons_mag_only": False,
        "apply_window_in_overlap": True,
        "mrstft_scales": [(256, 64, 256), (512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)],
        "lambda_mrstft": 0.75,
        "lambda_if_smooth": 0.1,
        "if_energy_weight": 1.5,
    },

    "amp": { "enabled": True, "dtype": "bfloat16", "force_float32_mps": False },

    "eval": { "val_every": 1, "val_batches": 100 },

    "log": { "log_every": 1000, "save_every": 1, "plot_every": 50000, "loss_json": "train_logs/loss_history.jsonl" },

    "train_opts": {
        "loss_log_interval": 10
    }
}

# Utilities
def _warn_to_tqdm(message, category, filename, lineno, file=None, line=None):
    log_rank0(f"[{category.__name__}] {message}", logger=tqdm.write)

warnings.showwarning = _warn_to_tqdm

def _device_and_scaler(device_override: Optional[torch.device] = None):
    dev = device_override
    if dev is None:
        dev = torch.device("mps" if torch.backends.mps.is_available()
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "mps" and CONFIG["amp"]["force_float32_mps"]:
        log_rank0("[train] Forzo float32 su MPS (no AMP)", logger=tqdm.write)
        return dev, None
    scaler = None
    if dev.type == "cuda" and CONFIG["amp"]["enabled"]:
        dtype_cfg = str(CONFIG["amp"].get("dtype", "float16")).lower()
        if dtype_cfg not in ("bf16", "bfloat16"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
    return dev, scaler

def _amp_autocast(dev: torch.device):
    if not CONFIG["amp"]["enabled"] or (dev.type == "mps" and CONFIG["amp"]["force_float32_mps"]):
        return nullcontext()
    if dev.type != "cuda":
        return nullcontext()
    dtype_cfg = str(CONFIG["amp"].get("dtype", "float16")).lower()
    if dtype_cfg in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)

def _ensure_dir(p: str | Path): Path(p).mkdir(parents=True, exist_ok=True)

def _load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def _roll_ctx(M_ctx, IF_ctx, M_tgt, IF_tgt, step, L):
    if step == 0: return M_ctx, IF_ctx
    return torch.cat([M_ctx, M_tgt[:, :step]], 1)[:, -L:], torch.cat([IF_ctx, IF_tgt[:, :step]], 1)[:, -L:]

def _cleanup(*tensors):
    for t in tensors: del t
    gc.collect()
    try: torch.mps.empty_cache()
    except: pass

def save_ckpt(path: Path, model: nn.Module, optimizer, scaler, epoch: int, global_step: int, stats: Dict, best_val: Optional[float] = None):
    _ensure_dir(path.parent)
    model_to_save = model.module if isinstance(model, DDP) else model
    torch.save({
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
        "epoch": epoch, "global_step": global_step, "stats": stats, "best_val": best_val
    }, path, _use_new_zipfile_serialization=False)


def _append_json_record(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _allowed_storage_roots() -> list[Path]:
    roots = []
    scratch = os.environ.get("CINECA_SCRATCH")
    work = os.environ.get("WORK")
    for val in (scratch, work):
        if val:
            try:
                roots.append(Path(val).resolve())
            except OSError:
                continue
    return roots

def _resolve_workspace_path(path_value: str | Path) -> str:
    if not path_value:
        return ""
    path_obj = Path(path_value)
    allowed_roots = _allowed_storage_roots()
    if not path_obj.is_absolute():
        if allowed_roots:
            return str((allowed_roots[0] / path_obj).resolve())
        return str(path_obj)
    if allowed_roots:
        resolved = path_obj.resolve()
        if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
            log_rank0(f"[paths] Warning: {resolved} fuori da CINECA_SCRATCH/WORK", logger=tqdm.write)
        return str(resolved)
    return str(path_obj)

def _resolve_config_paths():
    paths_cfg = CONFIG.get("paths", {})
    for key in ("train_data_root", "val_data_root", "manifest_train", "manifest_val", "stats_path", "ckpt_dir", "resume_ckpt"):
        if key in paths_cfg and paths_cfg[key]:
            paths_cfg[key] = _resolve_workspace_path(paths_cfg[key])

def _prepare_tmpdir():
    allowed_roots = _allowed_storage_roots()
    if not allowed_roots:
        tmpdir = os.environ.get("TMPDIR")
        if tmpdir:
            Path(tmpdir).mkdir(parents=True, exist_ok=True)
        return
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    desired = Path(allowed_roots[0]) / f"tmp_{job_id}"
    os.environ["TMPDIR"] = str(desired)
    desired.mkdir(parents=True, exist_ok=True)

class HybridConfig(dict):
    def __init__(self, d: dict): super().__init__(d); [setattr(self, k, v) for k, v in d.items()]

def build_model(F_bins: int, device: torch.device) -> IFFARModel:
    m = CONFIG["model"]
    enc_cfg = HybridConfig({
        "d_model": m["d_model"], "n_layers": m["n_layers"],
        "kernel_size": m["kernel_size"], "dropout": m["dropout"],
        "film_dim": m["flow_hidden"], "n_flow_layers": m["flow_layers"],
        "n_film_bands": m["n_film_bands"]
    })
    flow_cfg = HybridConfig({ "n_layers": m["flow_layers"], "hidden": m["flow_hidden"],
                              "kernel_size": m["flow_kernel_size"], "use_tanh_scale": m["flow_use_tanh_scale"],
                              "scale_factor": m["flow_scale_factor"] })
    mag_cfg = HybridConfig({ "d_model": m["d_model"], "hidden": m["mag_hidden"],
                             "use_prev_mag_skip": m["mag_use_prev_skip"], "spectral_smoothing": m["mag_spectral_smoothing"],
                             "kernel_size": m["mag_kernel"], "dropout": m["mag_dropout"],
                             "predict_logvar": m["mag_predict_logvar"] })
    recon_cfg = HybridConfig({ "n_fft": m["n_fft"], "hop_length": m["hop"], "win_length": m["win_length"],
                               "window": "hann", "center": m["center"], "return_waveform_in_chunk": True })
    return IFFARModel(enc_cfg, mag_cfg, flow_cfg, recon_cfg, F_bins=F_bins).to(device)

def build_loss() -> IFFARLoss:
    lc, m = CONFIG["loss"], CONFIG["model"]
    return IFFARLoss(n_fft=m["n_fft"], hop_length=m["hop"], win_length=m["win_length"], center_stft=m["center"],
                     lambda_if=lc["lambda_if"], lambda_if_reg=lc["lambda_if_reg"],
                     lambda_mag=lc["lambda_mag"], lambda_cons=lc["lambda_cons"],
                     lambda_overlap=lc["lambda_overlap"], lambda_time=lc["lambda_time"],
                     mag_loss=lc["mag_loss"], mag_huber_delta=lc["mag_huber_delta"],
                     time_alpha_sisdr=lc["time_alpha_sisdr"], time_beta_l1=lc["time_beta_l1"],
                     cons_mode=lc["cons_mode"], cons_mag_only=lc["cons_mag_only"],
                     apply_window_in_overlap=lc["apply_window_in_overlap"],
                     mrstft_scales=lc.get("mrstft_scales"),
                     lambda_mrstft=lc.get("lambda_mrstft", 0.0),
                     lambda_if_smooth=lc.get("lambda_if_smooth", 0.0),
                     if_energy_weight=lc.get("if_energy_weight", 1.0))

def build_optimizer(model: nn.Module): 
    oc = CONFIG["optim"]; return AdamW(model.parameters(), lr=oc["lr"], betas=oc["betas"], weight_decay=oc["weight_decay"])

def train():
    _resolve_config_paths()
    _prepare_tmpdir()

    dist_ctx = setup_distributed()
    dev_override = dist_ctx.device if dist_ctx.is_distributed else None
    dev, scaler = _device_and_scaler(dev_override)
    torch.set_float32_matmul_precision("high")

    per_gpu_batch = CONFIG["data"]["batch_size"]
    L = CONFIG["data"]["L"]
    K = CONFIG["data"]["K"]
    stride = CONFIG["data"]["stride"]
    world_size = dist_ctx.world_size
    global_batch = per_gpu_batch * world_size if dist_ctx.is_distributed else per_gpu_batch

    def _resolve_item_limit(value, label):
        if value is None or value is False:
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            log_rank0(f"[data] Ignoro {label} (valore non valido: {value})", logger=tqdm.write)
            return None
        if limit <= 0:
            log_rank0(f"[data] Ignoro {label} (valore <= 0: {limit})", logger=tqdm.write)
            return None
        return limit

    max_train_items = _resolve_item_limit(CONFIG["data"].get("max_train_items"), "max_train_items")
    max_val_items = _resolve_item_limit(CONFIG["data"].get("max_val_items"), "max_val_items")

    log_rank0(f"[setup] Distributed: {dist_ctx.is_distributed} (rank {dist_ctx.rank}/{world_size})", logger=tqdm.write)
    log_rank0(f"[setup] Device: {dev}, mixed precision: {CONFIG['amp']['dtype']}", logger=tqdm.write)
    log_rank0(f"[setup] Batch size per GPU: {per_gpu_batch}, global: {global_batch}, L={L}, K={K}", logger=tqdm.write)
    if is_rank0():
        if max_train_items is not None:
            log_rank0(f"[data] Uso max {max_train_items} finestre per il train", logger=tqdm.write)
        if max_val_items is not None:
            log_rank0(f"[data] Uso max {max_val_items} finestre per la val", logger=tqdm.write)

    P = CONFIG["paths"]
    stats = _load_json(P["stats_path"])
    mani_train = _load_json(P["manifest_train"])
    mani_val = _load_json(P["manifest_val"])
    F_bins = int(mani_train["F"])

    num_workers = CONFIG["data"]["num_workers"]
    worker_seed = int(CONFIG.get("train_opts", {}).get("worker_seed", 1234))
    pin_memory = dev.type == "cuda"
    val_batch_size = max(1, per_gpu_batch // 2)

    train_dataset = PackedWindowsDataset(
        manifest_path=P["manifest_train"],
        L=L,
        K=K,
        stride=stride,
        max_items_per_file=None,
        max_total_items=max_train_items,
        file_shuffle=True,
        use_half=True,
        seed=worker_seed,
    )
    val_dataset = PackedWindowsDataset(
        manifest_path=P["manifest_val"],
        L=L,
        K=K,
        stride=stride,
        max_items_per_file=None,
        max_total_items=max_val_items,
        file_shuffle=True,
        use_half=True,
        seed=worker_seed,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=dist_ctx.rank, shuffle=True, drop_last=True) if dist_ctx.is_distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=dist_ctx.rank, shuffle=False, drop_last=False) if dist_ctx.is_distributed else None

    def _make_loader(dataset, sampler, batch_size_loader, drop_last):
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size_loader,
            collate_fn=packed_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        else:
            loader_kwargs["shuffle"] = False
        if num_workers > 0:
            loader_kwargs["num_workers"] = num_workers
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
            loader_kwargs["worker_init_fn"] = lambda wid: _seed_worker(wid, worker_seed)
        else:
            loader_kwargs["num_workers"] = 0
        return DataLoader(**loader_kwargs)

    loader_train = _make_loader(train_dataset, train_sampler, per_gpu_batch, drop_last=True)
    loader_val = _make_loader(val_dataset, val_sampler, val_batch_size, drop_last=False)

    model = build_model(F_bins, dev)
    if dist_ctx.is_distributed:
        model = DDP(model, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank, broadcast_buffers=False)
    optimizer = build_optimizer(model)
    loss_fn = build_loss()

    train_opts = CONFIG.get("train_opts", {})
    flow_mean_mode_train = bool(train_opts.get("flow_mean_mode", True))
    flow_mean_prob = train_opts.get("flow_mean_prob")
    if flow_mean_prob is not None:
        try:
            flow_mean_prob = float(flow_mean_prob)
        except (TypeError, ValueError):
            flow_mean_prob = None
    compute_wave = bool(CONFIG["loss"].get("lambda_time", 0.0) > 0 or CONFIG["loss"].get("lambda_mrstft", 0.0) > 0)
    compute_wave_val = bool(train_opts.get("compute_wave_val", compute_wave))
    loss_log_interval = max(1, int(train_opts.get("loss_log_interval", 1)))

    loss_log_path_cfg = train_opts.get("loss_log_path") or CONFIG["log"].get("loss_json")
    collect_loss_stats = bool(loss_log_path_cfg)
    write_loss_records = collect_loss_stats and is_rank0()
    if write_loss_records:
        loss_log_path = Path(loss_log_path_cfg)
        loss_log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        loss_log_path = Path(loss_log_path_cfg) if collect_loss_stats else None

    run_id = train_opts.get("loss_run_id") or os.environ.get("SLURM_JOB_ID")
    if not run_id:
        run_id = f"run_{int(time.time())}" if dist_ctx.rank == 0 else ""
    if dist_ctx.is_distributed:
        obj_list = [run_id]
        dist.broadcast_object_list(obj_list, src=0)
        run_id = obj_list[0]

    global_step = 0
    best_val = None
    accum_steps = max(1, CONFIG["optim"]["grad_accum"])
    out_dir = Path(_resolve_workspace_path("train_plots"))
    if is_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(CONFIG["optim"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if is_rank0():
            pbar = tqdm(loader_train, desc=f"Train e{epoch+1}", dynamic_ncols=True, leave=False)
        else:
            pbar = None
        train_iter = pbar if pbar is not None else loader_train

        for it, batch in enumerate(train_iter):
            M_ctx = batch["M_ctx"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
            IF_ctx = batch["IF_ctx"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
            M_tgt = batch["M_tgt"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
            IF_tgt = batch["IF_tgt"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
            PHI_ctx = batch["phi_ctx"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
            PHI_tgt = batch["phi_tgt"].to(dev, dtype=torch.float32, non_blocking=pin_memory)

            phi0_chunk = PHI_ctx[:, -1]

            with _amp_autocast(dev):
                step_losses = []
                M_preds, IF_preds = [], []
                step_loss_sums = defaultdict(float) if collect_loss_stats else None
                if flow_mean_prob is not None and 0.0 <= flow_mean_prob <= 1.0:
                    mean_mode_flag = bool(np.random.random() < flow_mean_prob)
                else:
                    mean_mode_flag = flow_mean_mode_train
                for s in range(K):
                    M_ctx_s, IF_ctx_s = _roll_ctx(M_ctx, IF_ctx, M_tgt, IF_tgt, s, L)
                    if s == 0:
                        phi_ctx_s = PHI_ctx[:, -L:]
                    else:
                        phi_ctx_s = torch.cat([PHI_ctx, PHI_tgt[:, :s]], dim=1)[:, -L:]
                    phi_last = phi_ctx_s[:, -1]

                    out = model.forward_train(M_ctx_s, IF_ctx_s, M_tgt[:, s], IF_tgt[:, s], stats, phi_ctx_last=phi_last, mean_mode=mean_mode_flag)

                    losses = loss_fn(flow_nll=out["nll"], if_pred=out["IF_pred"], if_tgt=IF_tgt[:, s],
                                     m_pred=out["M_pred"], m_tgt=M_tgt[:, s], X_hat=out["X_hat"],
                                     y_hat=None, y_ref=None)
                    step_losses.append(losses["total"])
                    if step_loss_sums is not None:
                        for name, value in losses.items():
                            step_loss_sums[name] += float(value.detach())
                    M_preds.append(out["M_pred"].unsqueeze(1))
                    IF_preds.append(out["IF_pred"].unsqueeze(1))

                loss_total = sum(step_losses) / K

                M_pred_seq = torch.cat(M_preds, dim=1)
                IF_pred_seq = torch.cat(IF_preds, dim=1)
                X_pred_seq, phi_seq_pred, y_pred_seq = model.recon.reconstruct_chunk(M_pred_seq, IF_pred_seq, stats, return_waveform=compute_wave, phi0=phi0_chunk)
                with torch.no_grad():
                    _, phi_seq_gt, y_ref_seq = model.recon.reconstruct_chunk(M_tgt, IF_tgt, stats, return_waveform=compute_wave, phi0=phi0_chunk)

                overlap_term = None
                if phi_seq_pred is not None:
                    phi_for_loss = phi_seq_pred[:, 1:] if phi_seq_pred.size(1) == IF_pred_seq.size(1) + 1 else phi_seq_pred
                    overlap_term = loss_fn.lambda_overlap * loss_fn._loss_overlap_phase(phi_for_loss, IF_pred_seq)
                    loss_total = loss_total + overlap_term

                smooth_term = loss_fn.if_smooth_penalty(IF_pred_seq)
                loss_total = loss_total + smooth_term

                time_term = None
                mrstft_term = None
                if y_pred_seq is not None and y_ref_seq is not None and y_ref_seq.size(-1) > 0:
                    time_term = loss_fn.lambda_time * loss_fn._loss_time(y_pred_seq, y_ref_seq)
                    loss_total = loss_total + time_term
                    if loss_fn.lambda_mrstft > 0:
                        mrstft_term = loss_fn.lambda_mrstft * loss_fn._loss_mrstft(y_pred_seq, y_ref_seq)
                        loss_total = loss_total + mrstft_term

            if scaler:
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()

            if (it + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), CONFIG["optim"]["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), CONFIG["optim"]["max_grad_norm"])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if collect_loss_stats and (global_step + 1) % loss_log_interval == 0:
                loss_avg = {}
                if step_loss_sums:
                    for name, value in step_loss_sums.items():
                        loss_avg[name] = value / K
                loss_avg["overlap"] = float(overlap_term.detach()) if overlap_term is not None else 0.0
                loss_avg["if_smooth"] = float(smooth_term.detach())
                loss_avg["time"] = float(time_term.detach()) if time_term is not None else 0.0
                loss_avg["mrstft"] = float(mrstft_term.detach()) if mrstft_term is not None else 0.0
                loss_avg["total"] = float(loss_total.detach())
                if dist_ctx.is_distributed:
                    for name in list(loss_avg.keys()):
                        tensor = torch.tensor(loss_avg[name], device=dev)
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                        loss_avg[name] = tensor.item() / world_size
                if write_loss_records:
                    log_record = {
                        "run_id": run_id,
                        "split": "train",
                        "epoch": epoch + 1,
                        "batches": loss_log_interval,
                        "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    for name in sorted(loss_avg.keys()):
                        log_record[f"loss_{name}"] = loss_avg[name]
                    _append_json_record(loss_log_path, log_record)

            if is_rank0() and (global_step % CONFIG["log"]["plot_every"] == 0):
                pred_finite = torch.isfinite(out["M_pred"]).all()
                tgt_finite = torch.isfinite(M_tgt[:, 0]).all()
                if not (pred_finite and tgt_finite):
                    finite_info = {
                        "pred_finite": bool(pred_finite),
                        "tgt_finite": bool(tgt_finite),
                        "step": int(global_step),
                        "epoch": int(epoch + 1),
                    }
                    log_rank0(f"[plot] skip (non-finite values) -> {finite_info}", logger=tqdm.write)
                else:
                    M_pred = out["M_pred"].detach().float().cpu().numpy()
                    M_tgt0 = M_tgt[:, 0].detach().cpu().numpy()
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].imshow(20*np.log10(np.exp(M_tgt0)+1e-8)[0][None], aspect="auto", origin="lower", cmap="magma"); axes[0].set_title("Target dB")
                    axes[1].imshow(20*np.log10(np.exp(M_pred)+1e-8)[0][None], aspect="auto", origin="lower", cmap="magma"); axes[1].set_title("Pred dB")
                    out_path = out_dir / f"spec_step{global_step:06d}.png"
                    plt.savefig(out_path); plt.close(fig)
                    log_rank0(f"[plot] Saved {out_path}", logger=tqdm.write)

            _cleanup(M_ctx, IF_ctx, M_tgt, IF_tgt)
            global_step += 1

        if pbar is not None:
            pbar.close()

        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if (epoch + 1) % CONFIG["eval"]["val_every"] == 0:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            model.eval()
            val_loss_total, n_batches = 0.0, 0
            val_loss_sums_epoch = defaultdict(float) if collect_loss_stats else None
            with torch.no_grad():
                for it, batch in enumerate(loader_val):
                    M_ctx = batch["M_ctx"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
                    IF_ctx = batch["IF_ctx"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
                    M_tgt = batch["M_tgt"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
                    IF_tgt = batch["IF_tgt"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
                    PHI_ctx = batch["phi_ctx"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
                    PHI_tgt = batch["phi_tgt"].to(dev, dtype=torch.float32, non_blocking=pin_memory)
                    phi0_chunk = PHI_ctx[:, -1]
                    with _amp_autocast(dev):
                        step_losses = []
                        M_preds, IF_preds = [], []
                        step_loss_sums = defaultdict(float) if collect_loss_stats else None
                        for s in range(K):
                            M_ctx_s, IF_ctx_s = _roll_ctx(M_ctx, IF_ctx, M_tgt, IF_tgt, s, L)
                            if s == 0:
                                phi_ctx_s = PHI_ctx[:, -L:]
                            else:
                                phi_ctx_s = torch.cat([PHI_ctx, PHI_tgt[:, :s]], dim=1)[:, -L:]
                            phi_last = phi_ctx_s[:, -1]

                            out = model.forward_train(M_ctx_s, IF_ctx_s, M_tgt[:, s], IF_tgt[:, s], stats, phi_ctx_last=phi_last, mean_mode=True)

                            losses = loss_fn(flow_nll=out["nll"], if_pred=out["IF_pred"], if_tgt=IF_tgt[:, s],
                                             m_pred=out["M_pred"], m_tgt=M_tgt[:, s], X_hat=out["X_hat"],
                                             y_hat=None, y_ref=None)
                            step_losses.append(losses["total"])
                            if step_loss_sums is not None:
                                for name, value in losses.items():
                                    step_loss_sums[name] += float(value.detach())
                            M_preds.append(out["M_pred"].unsqueeze(1))
                            IF_preds.append(out["IF_pred"].unsqueeze(1))

                        loss_total = sum(step_losses) / K

                        M_pred_seq = torch.cat(M_preds, dim=1)
                        IF_pred_seq = torch.cat(IF_preds, dim=1)
                        X_pred_seq, phi_seq_pred, y_pred_seq = model.recon.reconstruct_chunk(M_pred_seq, IF_pred_seq, stats, return_waveform=compute_wave_val, phi0=phi0_chunk)
                        with torch.no_grad():
                            _, phi_seq_gt, y_ref_seq = model.recon.reconstruct_chunk(M_tgt, IF_tgt, stats, return_waveform=compute_wave_val, phi0=phi0_chunk)

                        overlap_term = None
                        if phi_seq_pred is not None:
                            phi_for_loss = phi_seq_pred[:, 1:] if phi_seq_pred.size(1) == IF_pred_seq.size(1) + 1 else phi_seq_pred
                            overlap_term = loss_fn.lambda_overlap * loss_fn._loss_overlap_phase(phi_for_loss, IF_pred_seq)
                            loss_total = loss_total + overlap_term

                        smooth_term = loss_fn.if_smooth_penalty(IF_pred_seq)
                        loss_total = loss_total + smooth_term

                        time_term = None
                        mrstft_term = None
                        if y_pred_seq is not None and y_ref_seq is not None and y_ref_seq.size(-1) > 0:
                            time_term = loss_fn.lambda_time * loss_fn._loss_time(y_pred_seq, y_ref_seq)
                            loss_total = loss_total + time_term
                            if loss_fn.lambda_mrstft > 0:
                                mrstft_term = loss_fn.lambda_mrstft * loss_fn._loss_mrstft(y_pred_seq, y_ref_seq)
                                loss_total = loss_total + mrstft_term

                        loss_scalar = float(loss_total.detach())
                        val_loss_total += loss_scalar
                        n_batches += 1

                        if val_loss_sums_epoch is not None:
                            loss_avg = {}
                            if step_loss_sums:
                                for name, value in step_loss_sums.items():
                                    loss_avg[name] = value / K
                            loss_avg["overlap"] = float(overlap_term.detach()) if overlap_term is not None else 0.0
                            loss_avg["if_smooth"] = float(smooth_term.detach())
                            loss_avg["time"] = float(time_term.detach()) if time_term is not None else 0.0
                            loss_avg["mrstft"] = float(mrstft_term.detach()) if mrstft_term is not None else 0.0
                            loss_avg["total"] = loss_scalar

                            for name, value in loss_avg.items():
                                val_loss_sums_epoch[name] += value

            loss_stats_tensor = torch.tensor([val_loss_total, float(n_batches)], device=dev)
            if dist_ctx.is_distributed:
                dist.all_reduce(loss_stats_tensor, op=dist.ReduceOp.SUM)
            total_batches = max(1, int(loss_stats_tensor[1].item()))
            val_loss_mean = loss_stats_tensor[0].item() / total_batches if total_batches > 0 else 0.0
            log_rank0(f"[epoch {epoch+1}] val_total={val_loss_mean:.4f}", logger=tqdm.write)

            improved = best_val is None or val_loss_mean < best_val
            if improved:
                best_val = val_loss_mean

            if improved and is_rank0():
                save_ckpt(Path(P["ckpt_dir"]) / "best.pt", model, optimizer, scaler, epoch + 1, global_step, stats, best_val)
                log_rank0("[ckpt] saved BEST", logger=tqdm.write)

            if collect_loss_stats and val_loss_sums_epoch is not None and total_batches > 0:
                val_means = {}
                for name, value in val_loss_sums_epoch.items():
                    tensor = torch.tensor(value, device=dev)
                    if dist_ctx.is_distributed:
                        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    val_means[name] = tensor.item() / total_batches
                if write_loss_records:
                    log_record = {
                        "run_id": run_id,
                        "split": "val",
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "batches": total_batches,
                    }
                    for name in sorted(val_means.keys()):
                        log_record[f"loss_{name}"] = val_means[name]
                    _append_json_record(loss_log_path, log_record)

        if is_rank0():
            save_ckpt(Path(P["ckpt_dir"]) / f"epoch_{epoch+1:03d}.pt", model, optimizer, scaler, epoch + 1, global_step, stats, best_val)
            log_rank0("[ckpt] saved epoch", logger=tqdm.write)

    if is_rank0():
        save_ckpt(Path(P["ckpt_dir"]) / "last.pt", model, optimizer, scaler, CONFIG["optim"]["epochs"], global_step, stats, best_val)
    barrier()
    log_rank0("[train] done.", logger=tqdm.write)
    cleanup_distributed()


if __name__ == "__main__":
    train()
