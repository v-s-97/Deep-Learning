from __future__ import annotations
import torch
import torch.nn as nn

from model.tcn_encoder import TCNContextEncoder
from model.mag_head import MagHead
from model.if_flow import IFConditionalFlow, IFFlowConfig
from model.phase_rec import PhaseReconstructor


class IFFARModel(nn.Module):
    def __init__(self,
                 enc_cfg,
                 mag_cfg,
                 flow_cfg,
                 recon_cfg,
                 F_bins: int):
        super().__init__()

        self.encoder = TCNContextEncoder(enc_cfg, F_bins=F_bins)

        self.mag_head = MagHead(
            d_model=mag_cfg.d_model,
            hidden=mag_cfg.hidden,
            kernel_size=mag_cfg.kernel_size,
            dropout=mag_cfg.dropout,
            use_prev_mag_skip=mag_cfg.use_prev_mag_skip,
            predict_logvar=mag_cfg.predict_logvar,
            spectral_smoothing=mag_cfg.spectral_smoothing
        )

        flow_cfg_obj = IFFlowConfig(
            n_layers=flow_cfg.n_layers,
            hidden=flow_cfg.hidden,
            kernel_size=flow_cfg.kernel_size,
            use_tanh_scale=flow_cfg.use_tanh_scale,
            scale_factor=flow_cfg.scale_factor,
        )
        self.if_flow = IFConditionalFlow(F_bins=F_bins, cfg=flow_cfg_obj)

        get = recon_cfg.get if hasattr(recon_cfg, "get") else lambda k, d=None: getattr(recon_cfg, k, d)
        self.recon = PhaseReconstructor(
            n_fft=get("n_fft"),
            hop_length=get("hop_length"),
            win_length=get("win_length"),
            window=get("window", "hann"),
            center=get("center", True),
            return_waveform_in_chunk=get("return_waveform_in_chunk", True)
        )

        self.F_bins = F_bins

    def forward_train(self,
                      M_ctx,
                      IF_ctx,
                      M_target,
                      IF_target,
                      stats,
                      *,
                      phi_ctx_last: torch.Tensor | None = None,
                      mean_mode: bool = True,
                      reconstruct_waveform: bool = False):
        """Forward pass su un singolo step autoregressivo."""
        C, film_params = self.encoder(M_ctx, IF_ctx)

        prev_mag = M_ctx[:, -1] if getattr(self.mag_head, "use_prev_mag_skip", False) else None
        mag_out = self.mag_head(C, prev_mag=prev_mag)
        M_pred = mag_out["mag"]

        IF_pred, nll = self.if_flow(film_params, y_target=IF_target, mean_mode=mean_mode)

        X_hat, phi, y_hat = self.recon.reconstruct_chunk(
            M_pred, IF_pred, stats, return_waveform=reconstruct_waveform, phi0=phi_ctx_last
        )

        return {
            "M_pred": M_pred,
            "IF_pred": IF_pred,
            "nll": nll,
            "X_hat": X_hat,
            "phi": phi,
            "y_hat": y_hat,
            "mag_aux": mag_out,
        }

    def forward(self,
                M_ctx,
                IF_ctx,
                M_target,
                IF_target,
                stats,
                *,
                phi_ctx_last: torch.Tensor | None = None,
                mean_mode: bool = True,
                reconstruct_waveform: bool = False):
        return self.forward_train(
            M_ctx,
            IF_ctx,
            M_target,
            IF_target,
            stats,
            phi_ctx_last=phi_ctx_last,
            mean_mode=mean_mode,
            reconstruct_waveform=reconstruct_waveform,
        )

    @torch.no_grad()
    def forward_eval(self,
                     M_ctx: torch.Tensor,
                     IF_ctx: torch.Tensor,
                     stats: dict,
                     K: int,
                     L: int,
                     phi_ctx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        was_training = self.training
        if was_training:
            self.eval()
        ctx_mag = M_ctx.clone()
        ctx_if = IF_ctx.clone()
        ctx_phi = phi_ctx.clone() if phi_ctx is not None else None
        phi0_chunk = ctx_phi[:, -1] if ctx_phi is not None else None
        mags, ifs = [], []
        prev_mag = ctx_mag[:, -1] if getattr(self.mag_head, "use_prev_mag_skip", False) else None

        for _ in range(K):
            C, film_params = self.encoder(ctx_mag, ctx_if)
            mag_out = self.mag_head(C, prev_mag=prev_mag)
            mag_pred = mag_out["mag"]
            if_pred, _ = self.if_flow(film_params, mean_mode=True)

            mags.append(mag_pred.unsqueeze(1))
            ifs.append(if_pred.unsqueeze(1))

            ctx_mag = torch.cat([ctx_mag, mag_pred.unsqueeze(1)], dim=1)[:, -L:]
            ctx_if = torch.cat([ctx_if, if_pred.unsqueeze(1)], dim=1)[:, -L:]
            if ctx_phi is not None:
                phi_last = ctx_phi[:, -1]
                IF_pred_denorm = PhaseReconstructor.denorm_if(if_pred.unsqueeze(1), stats).squeeze(1)
                phi_next = (phi_last + IF_pred_denorm).unsqueeze(1)
                ctx_phi = torch.cat([ctx_phi, phi_next], dim=1)[:, -L:]
            prev_mag = mag_pred if getattr(self.mag_head, "use_prev_mag_skip", False) else None

        M_pred_seq = torch.cat(mags, dim=1)
        IF_pred_seq = torch.cat(ifs, dim=1)
        if was_training:
            self.train()
        return M_pred_seq, IF_pred_seq, phi0_chunk
