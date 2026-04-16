# ================================================================
# model.py
#   - MomentPredictor: predicts moments in normalized (z-score) space
#   - PhysicsLoss: compares reconstructed FSP bins + moments + Vtot
#     with Kendall uncertainty-weighted multi-task loss.
# ================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fsp_c import FSPReconstructor, vb
from utils import bin_centers, calc_Vtot_batched

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MODEL] Using device: {device}")


# ================================================================
# Positional Encoding (for Transformer)
# ================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ================================================================
# MomentPredictor (NOW CAUSAL AUTOREGRESSIVE)
# ================================================================
class MomentPredictor(nn.Module):

    def __init__(
        self,
        input_dim_scalar: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        max_timesteps: int = 15,
        model_type: str = "transformer",
        output_mean=None,
        output_std=None,
        scheduled_sampling_start=1.0,
        scheduled_sampling_end=0.2,
        scheduled_sampling_decay=1e-4,
    ):
        super().__init__()

        self.model_type = model_type.lower()
        self.hidden_dim = hidden_dim
        self.max_timesteps = max_timesteps

        # ===== AUTOREGRESSIVE ADDITION =====
        self.teacher_forcing_prob = scheduled_sampling_start
        self.tf_end = scheduled_sampling_end
        self.tf_decay = scheduled_sampling_decay
        self.moment_start_idx = -3  # assumes last 3 scalar inputs are moments

        assert output_mean is not None and output_std is not None
        self.register_buffer("moment_mean",
                             torch.tensor(output_mean, dtype=torch.float32).view(1,1,-1))
        self.register_buffer("moment_std",
                             torch.tensor(output_std, dtype=torch.float32).view(1,1,-1))

        self.input_proj = nn.Linear(input_dim_scalar, hidden_dim)

        if self.model_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.backbone = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_timesteps)
        else:
            raise ValueError("Use transformer for autoregressive mode.")

        self.moment_head = nn.Linear(hidden_dim, 3)
        self.fsp = FSPReconstructor(vb).to(device)

    # ------------------------------------------------------------
    # Causal Mask
    # ------------------------------------------------------------
    def _generate_causal_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        return mask.to(device)

    # ------------------------------------------------------------
    # Scheduled sampling update (call automatically)
    # ------------------------------------------------------------
    def _update_teacher_forcing(self):
        if self.training:
            self.teacher_forcing_prob = max(
                self.tf_end,
                self.teacher_forcing_prob - self.tf_decay
            )

    # ------------------------------------------------------------
    # Inverse normalization
    # ------------------------------------------------------------
    def moments_to_physical(self, y_norm):
        return y_norm * self.moment_std + self.moment_mean

    def forward_shap(self, x):
        # NO autoregression, full sequence
        h = self.input_proj(x)
        h = self.pos_encoding(h)

        causal_mask = self._generate_causal_mask(x.shape[1], x.device)
        h = self.backbone(h, mask=causal_mask)

        return self.moment_head(h)  # (B, T, 3)
    # ------------------------------------------------------------
    # AUTOREGRESSIVE FORWARD
    # ------------------------------------------------------------
    def forward(self, x_scalar_norm):

        B, T, D = x_scalar_norm.shape
        device = x_scalar_norm.device

        preds = []
        x_roll = x_scalar_norm.clone()

        for t in range(T):

            x_partial = x_roll[:, :t + 1, :]

            h = self.input_proj(x_partial)
            h = self.pos_encoding(h)

            causal_mask = self._generate_causal_mask(t + 1, device)
            h = self.backbone(h, mask=causal_mask)

            y_t = self.moment_head(h[:, -1, :])
            preds.append(y_t.unsqueeze(1))

            # scheduled sampling update
            if t < T - 1:

                use_teacher = (
                    torch.rand(1).item() < self.teacher_forcing_prob
                    if self.training else False
                )

                x_next = x_roll[:, t + 1, :].clone()
                # use_teacher must be true for shap analysis
                #use_teacher = True
                # this is only for SHAP
                #x_next[:, self.moment_start_idx:] = y_t

                if use_teacher:
                    x_next[:, self.moment_start_idx:] = \
                        x_scalar_norm[:, t + 1, self.moment_start_idx:]
                else:
                    x_next[:, self.moment_start_idx:] = y_t.detach()

                x_roll = torch.cat([
                    x_roll[:, :t + 1, :],
                    x_next.unsqueeze(1),
                    x_roll[:, t + 2:, :]
                ], dim=1)

        preds = torch.cat(preds, dim=1)
        self._update_teacher_forcing()

        return preds

    # ------------------------------------------------------------
    # FSP Reconstruction unchanged
    # ------------------------------------------------------------
    def reconstruct_bins_from_norm(self, y_norm):
        y_phys = self.moments_to_physical(y_norm)
        logN   = y_phys[..., 0]
        logCMD = y_phys[..., 1]
        GSD    = y_phys[..., 2]

        B, T = logN.shape

        logN_flat   = logN.reshape(-1)
        logCMD_flat = logCMD.reshape(-1)
        GSD_flat    = GSD.reshape(-1)

        with torch.no_grad():
            bins_all = self.fsp(
                log10_Ntot=logN_flat,
                log10_CMD_nm=logCMD_flat,
                GSD_linear=GSD_flat,
            )

        bins_all = bins_all[:, 1:]
        bins_phys = bins_all.reshape(B, T, -1)

        return bins_phys


# ================================================================
# MomentPredictor
#   - Inputs: scalar features only (normalized)
#   - Outputs: normalized moments (z-score)
#     [log10(N_tot), log10(CMD_nm), GSD]
# ================================================================
class MomentPredictor_seq(nn.Module):
    def __init__(
        self,
        input_dim_scalar: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        max_timesteps: int = 15,
        model_type: str = "transformer",
        output_mean=None,
        output_std=None,
    ):
        """
        Parameters
        ----------
        input_dim_scalar : int
            Number of scalar input features per timestep (already normalized).
        hidden_dim : int
            Hidden dimension of the backbone model.
        num_layers : int
            Number of Transformer or RNN layers.
        dropout : float
            Dropout probability.
        nhead : int
            Number of attention heads for Transformer.
        max_timesteps : int
            Maximum sequence length (for positional encoding).
        model_type : str
            "transformer", "lstm", or "gru".
        output_mean, output_std : array-like of shape (3,)
            Mean and std used to normalize the 3 moments
            [log10(N_tot), log10(CMD_nm), GSD].
        """
        super().__init__()

        self.model_type = model_type.lower()
        self.input_dim_scalar = input_dim_scalar
        self.hidden_dim = hidden_dim
        self.max_timesteps = max_timesteps

        # Store moment normalization statistics for inverse transform
        assert output_mean is not None and output_std is not None, \
            "output_mean and output_std from dataset scaler are required."
        self.register_buffer("moment_mean", torch.tensor(output_mean, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("moment_std",  torch.tensor(output_std,  dtype=torch.float32).view(1, 1, -1))

        # Input projection
        self.input_proj = nn.Linear(input_dim_scalar, hidden_dim)

        # Backbone
        if self.model_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.backbone = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_timesteps)

        elif self.model_type in ["lstm", "gru"]:
            rnn_cls = nn.LSTM if self.model_type == "lstm" else nn.GRU
            self.backbone = rnn_cls(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
            self.pos_encoding = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Output head: 3 normalized moments per timestep
        self.moment_head = nn.Linear(hidden_dim, 3)

        # FSP reconstructor (fixed, not trained)
        self.fsp = FSPReconstructor(vb).to(device)

    # ------------------------------------------------------------
    # Helpers for normalization
    # ------------------------------------------------------------
    def moments_to_physical(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized (z-score) moments to physical moment space.

        y_norm: (B, T, 3)
        returns: (B, T, 3)
          [log10(N_tot), log10(CMD_nm), GSD]
        """
        return y_norm * self.moment_std + self.moment_mean

    # ------------------------------------------------------------
    # Forward: predict normalized moments
    # ------------------------------------------------------------
    def forward(self, x_scalar_norm: torch.Tensor) -> torch.Tensor:
        """
        x_scalar_norm: (B, T, input_dim_scalar)  -- already normalized by dataset
        returns:
            y_norm: (B, T, 3) normalized moments
        """
        h = self.input_proj(x_scalar_norm)

        if self.model_type == "transformer":
            h = self.pos_encoding(h)
            h = self.backbone(h)
        else:
            h, _ = self.backbone(h)

        y_norm = self.moment_head(h)  # (B, T, 3)
        return y_norm

    # ------------------------------------------------------------
    # Reconstruct FSP bins from normalized moments
    # ------------------------------------------------------------
    def reconstruct_bins_from_norm(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Given normalized moment predictions, reconstruct physical Nconc bins.

        y_norm: (B, T, 3) normalized [log10N, log10CMD_nm, GSD]
        returns:
            bins_phys: (B, T, 57)   physical Nconc per bin
        """
        # Convert to physical log10 moments & GSD
        y_phys = self.moments_to_physical(y_norm)           # (B, T, 3)
        logN   = y_phys[..., 0]                             # (B, T)
        logCMD = y_phys[..., 1]                             # (B, T)
        GSD    = y_phys[..., 2]                             # (B, T)

        B, T = logN.shape
        # Flatten batch + time so FSP runs in one shot
        logN_flat   = logN.reshape(-1)                      # (B*T,)
        logCMD_flat = logCMD.reshape(-1)                    # (B*T,)
        GSD_flat    = GSD.reshape(-1)                       # (B*T,)

        with torch.no_grad():
            bins_all = self.fsp(
                log10_Ntot=logN_flat,
                log10_CMD_nm=logCMD_flat,
                GSD_linear=GSD_flat,
            )                                              # (B*T, 58) with dummy bin0

        bins_all = bins_all[:, 1:]                         # drop dummy bin 0 → (B*T, 57)
        bins_phys = bins_all.reshape(B, T, -1)             # (B, T, 57)

        return bins_phys


# ================================================================
# PhysicsLoss (Kendall uncertainty-weighted)
#   - Works in physical space
#   - Takes normalized moments and physical bins
# ================================================================
class PhysicsLoss(nn.Module):
    def __init__(self, n_bins, output_mean, output_std, fsp,
                 mom_weight=1.0, bin_weight=1e-4, vtot_weight=1.0,
                 use_uncertainty=True):
        super().__init__()

        # Convert weights to floats (important!)
        self.fsp = fsp.to(output_mean.device if hasattr(output_mean, 'device') else 'cpu')
        self.mom_weight  = float(mom_weight)
        self.bin_weight  = float(bin_weight)
        self.vtot_weight = float(vtot_weight)

        # Store normalization stats
        self.register_buffer("moment_mean",
                             torch.tensor(output_mean, dtype=torch.float32).view(1,1,-1))
        self.register_buffer("moment_std",
                             torch.tensor(output_std, dtype=torch.float32).view(1,1,-1))

        # Bin centers (will move automatically to GPU)
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32))

        # Uncertainty parameters
        if use_uncertainty:
            self.log_sigma_mom  = nn.Parameter(torch.zeros(1))
            self.log_sigma_bins = nn.Parameter(torch.zeros(1))
            self.log_sigma_vtot = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_sigma_mom",  torch.tensor(0.0))
            self.register_buffer("log_sigma_bins", torch.tensor(0.0))
            self.register_buffer("log_sigma_vtot", torch.tensor(0.0))

    # Restore physical scale for moments
    def moments_to_physical(self, y_norm):
        return y_norm * self.moment_std + self.moment_mean

    def forward(
            self,
            pred_mom_norm: torch.Tensor,  # (B,T,3)
            true_mom_norm: torch.Tensor,  # (B,T,3)
            true_bins: torch.Tensor,  # (B,T,57)
            true_vtot: torch.Tensor = None,
            debug=True,
            epoch=0
    ):
        """
        pred_mom_norm : normalized predicted [log10N, log10CMD, GSD]
        true_mom_norm : normalized true [log10N, log10CMD, GSD]
        true_bins     : physical FSP bins from dataset
        """

        B, T, _ = pred_mom_norm.shape

        # -----------------------------------------------------
        # 1) Inverse normalization → physical moment space
        # -----------------------------------------------------
        pred_mom_phys = self.moments_to_physical(pred_mom_norm)
        true_mom_phys = self.moments_to_physical(true_mom_norm)

        # Extract components
        logN = pred_mom_phys[..., 0].reshape(-1)
        logCMD = pred_mom_phys[..., 1].reshape(-1)
        GSD = pred_mom_phys[..., 2].reshape(-1)

        # -----------------------------------------------------
        # 2) Stabilize ranges before FSP
        # -----------------------------------------------------
        logN = torch.clamp(logN, min=-5.0, max=20.0)
        logCMD = torch.clamp(logCMD, min=-1.0, max=7.0)
        GSD = torch.clamp(GSD, min=1.01, max=5.0)

        # -----------------------------------------------------
        # 3) Reconstruct bins using FSP
        # -----------------------------------------------------
        bins_all = self.fsp(
            log10_Ntot=logN,
            log10_CMD_nm=logCMD,
            GSD_linear=GSD,
        )
        pred_bins = bins_all[:, 1:].reshape(B, T, 57)  # drop dummy bin
        # pred_bins = pred_bins.detach()
        # -----------------------------------------------------
        # 4) Compute losses (log-bin MSE, moment MSE, vtot MSE); FIRST CLAMP IT
        # -----------------------------------------------------
        eps = 1e-30
        pred_bins_c = torch.clamp(pred_bins, min=eps)
        true_bins_c = torch.clamp(true_bins, min=eps)

        # nan_pred_bins_c = torch.isnan(pred_bins_c).any()
        #if nan_pred_bins_c:
        #    print("NAN In pred bins")
        #else:
        #    print("NO NAN In pred bins")

        # Moments MSE
        loss_mom = F.mse_loss(pred_mom_phys, true_mom_phys)

        if epoch <= 0 and self.bin_weight == 0.0:

            loss_bins = pred_mom_norm.new_tensor(0.0)
            effective_bin_weight = 0

        elif epoch <= 0 and self.bin_weight > 0.0:
            pred_bins_detached = pred_bins.detach()
            loss_bins = pred_mom_norm.new_tensor(0.0)
            effective_bin_weight = 0


        else:
            if self.bin_weight == 0.0:
                loss_bins = pred_mom_norm.new_tensor(0.0)
                effective_bin_weight = 0
            else:
                # Bin loss in log10 space
                #pred_bins_c = torch.clamp(pred_bins, min=1e-30, max=1e30)
                #true_bins_c = torch.clamp(pred_bins, min=1e-30, max=1e30)
                #print(pred_bins_c, true_bins_c)
                #log_ratio = torch.log(pred_bins_c / true_bins_c)
                #loss_bins = torch.mean(log_ratio ** 2)
                #nan_pred_bins_c = torch.isnan(pred_bins_c).any()
                #nan_true_bins_c = torch.isnan(true_bins_c).any()
                #if nan_pred_bins_c:
                    #print("NAN In pred bins")
                    #with torch.set_printoptions(threshold=float('inf')):
                #    print(pred_bins_c)

                #else:
                #    print("NO NAN In pred bins")

                #if nan_true_bins_c:
                #    print("NAN In true bins")

                #else:
                #    print("NO NAN In true bins")

                loss_bins = F.mse_loss(torch.log10(pred_bins_c), torch.log10(true_bins_c))
                effective_bin_weight = self.bin_weight
                # print(epoch, loss_bins, self.bin_weight)

        #print('outer:', loss_bins)
        if self.vtot_weight == 0.0:
            loss_vtot = pred_mom_norm.new_tensor(0.0)
        else:
            # Vtot loss
            pred_vtot = calc_Vtot_batched(pred_bins_c, self.bin_centers)
            true_vtot = true_vtot if true_vtot is not None else calc_Vtot_batched(true_bins_c, self.bin_centers)
            loss_vtot = F.mse_loss(pred_vtot, true_vtot)

        # -----------------------------------------------------
        # 5) Kendall uncertainty weighting
        # -----------------------------------------------------
        inv_mom = torch.exp(-2 * self.log_sigma_mom)
        inv_bins = torch.exp(-2 * self.log_sigma_bins)
        inv_vtot = torch.exp(-2 * self.log_sigma_vtot)

        total = (
                self.mom_weight * inv_mom * loss_mom +
                effective_bin_weight * inv_bins * loss_bins +
                self.vtot_weight * inv_vtot * loss_vtot +
                (self.log_sigma_mom + self.log_sigma_bins + self.log_sigma_vtot)
        )
        #print(epoch, loss_bins)
        if debug:
            return total, {
                "loss_mom": loss_mom.detach(),
                "loss_bins": loss_bins.detach(),
                "loss_vtot": loss_vtot.detach(),
                "total": total.detach()
            }, pred_bins, pred_mom_phys
        else:
            return total, {
                "loss_mom": loss_mom.detach(),
                "loss_bins": loss_bins.detach(),
                "loss_vtot": loss_vtot.detach(),
                "total": total.detach()
            }

