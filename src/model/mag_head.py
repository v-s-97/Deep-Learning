from __future__ import annotations
import torch
import torch.nn as nn

class MagHead(nn.Module):
    def __init__(self,
                 d_model: int,
                 hidden: int,
                 kernel_size: int = 5,
                 dropout: float = 0.1,
                 use_prev_mag_skip: bool = False,
                 spectral_smoothing: bool = False,
                 predict_logvar: bool = False):
        super().__init__()

        self.use_prev_mag_skip = use_prev_mag_skip
        self.spectral_smoothing = spectral_smoothing
        self.predict_logvar = predict_logvar

        self.fc1 = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.fc_out = nn.Linear(hidden, 1)

        if predict_logvar:
            self.fc_logvar = nn.Linear(hidden, 1)

        if spectral_smoothing:
            self.conv_smooth = nn.Conv1d(1, 1, kernel_size,
                                         padding=kernel_size // 2)

    def forward(self,
                h: torch.Tensor,
                prev_mag: torch.Tensor | None = None) -> dict:
        """
        h: [B,F,d_model] hidden features dall'encoder
        prev_mag: [B,F] magnitudo precedente (se skip abilitato)
        """
        B, F, _ = h.shape

        z = self.fc1(h)
        z = self.act(z)
        z = self.dropout(z)

        mag_pred = self.fc_out(z).squeeze(-1)

        if self.use_prev_mag_skip and prev_mag is not None:
            mag_pred = mag_pred + prev_mag

        if self.spectral_smoothing:
            mag_pred = mag_pred.unsqueeze(1)
            mag_pred = self.conv_smooth(mag_pred)
            mag_pred = mag_pred.squeeze(1)

        out = {"mag": mag_pred}

        if self.predict_logvar:
            logvar = self.fc_logvar(z).squeeze(-1)
            out["logvar"] = logvar

        return out
