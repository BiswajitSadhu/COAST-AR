import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from fsp_c import FSPReconstructor, vb
from utils import bin_centers, calc_Vtot_batched

class AerosolSimDataset(Dataset):
    def __init__(
        self,
        csv_file,
        sim_len=15,
        fit_scaler=True,
        scaler_stats=None,
        log_transform=True
    ):
        """
        Dataset for physics-informed aerosol moment prediction.
        Model DOES NOT receive FSP bins as input.
        FSP bins & Vtot ARE returned for the loss function.
        """
        self.sim_len = sim_len
        self.log_transform = log_transform

        df = pd.read_csv(csv_file)

        # ====================================================
        # INPUT COLUMNS (what model sees)
        # ====================================================
        input_cols_scalar = [
            "Temperature",
            "density",
            "Pressure",
            "time_tao_ratio",

            "N_tot",     # will be log10 transformed
            "CMD_nm",    # will be log10 transformed
            "GSD",       # NOT log transformed
        ]

        # ====================================================
        # TARGET: model predicts moments
        # ====================================================
        target_cols = ["N_tot", "CMD_nm", "GSD"]

        # ====================================================
        # TRUE distribution & Vtot for physics loss
        # ====================================================
        bin_cols = [f"Nconc_fsp_bin_{i:02d}" for i in range(57)]
        assert all(col in df.columns for col in bin_cols)

        if "Vtot_from_nconc" not in df.columns:
            raise ValueError("Dataset must contain Vtot_from_nconc column")

        # ====================================================
        # LOG10 transform ONLY for Ntot & CMD_nm
        # ====================================================
        eps = 1e-30
        if log_transform:
            df["N_tot"]  = np.log10(df["N_tot"].clip(lower=eps))
            df["CMD_nm"] = np.log10(df["CMD_nm"].clip(lower=eps))
            # GSD stays linear
            print("Applied log10 transform to N_tot and CMD_nm ONLY.")

        # ====================================================
        # Prepare arrays
        # ====================================================
        X_scalar = df[input_cols_scalar].to_numpy(np.float32)
        Y = df[target_cols].to_numpy(np.float32)
        true_bins = df[bin_cols].to_numpy(np.float32)
        true_vtot = df["Vtot_from_nconc"].to_numpy(np.float32)

        assert len(df) % sim_len == 0
        self.n_sims = len(df) // sim_len

        # Save simulation index per simulation (length = n_sims)
        self.sim_index_values = df["sim_index"].values[::sim_len]

        # reshape
        X_scalar = X_scalar.reshape(self.n_sims, sim_len, -1)
        Y        = Y.reshape(self.n_sims, sim_len, -1)
        true_bins = true_bins.reshape(self.n_sims, sim_len, 57)
        true_vtot = true_vtot.reshape(self.n_sims, sim_len)

        # ====================================================
        # Fit normalization on MODEL INPUTS + TARGETS ONLY
        # (true bins & vtot are NOT normalized)
        # ====================================================
        if fit_scaler:
            self.scalar_mean = X_scalar.mean((0,1))
            self.scalar_std  = X_scalar.std((0,1)) + 1e-8

            self.output_mean = Y.mean((0,1))
            self.output_std  = Y.std((0,1)) + 1e-8
        else:
            self.scalar_mean = scaler_stats["scalar_mean"]
            self.scalar_std  = scaler_stats["scalar_std"]
            self.output_mean = scaler_stats["output_mean"]
            self.output_std  = scaler_stats["output_std"]

        # normalize scalar inputs + target moments
        X_scalar_norm = (X_scalar - self.scalar_mean) / self.scalar_std
        Y_norm        = (Y - self.output_mean) / self.output_std

        # ====================================================
        # Save tensors
        # ====================================================
        self.inputs_scalar = torch.from_numpy(X_scalar_norm).float()
        self.targets_moments = torch.from_numpy(Y_norm).float()
        self.true_bins = torch.from_numpy(true_bins).float()
        self.true_vtot = torch.from_numpy(true_vtot).float()

    # ====================================================
    # PyTorch API
    # ====================================================
    def __len__(self):
        return self.n_sims

    def __getitem__(self, idx):
        return (
            self.inputs_scalar[idx],     # (sim_len, scalar_features)
            self.targets_moments[idx],   # (sim_len, 3)
            self.true_bins[idx],         # (sim_len, 57)
            self.true_vtot[idx],         # (sim_len,)
        )

    # ====================================================
    # Scaler utilities
    # ====================================================
    def get_scaler_stats(self):
        return {
            "scalar_mean": self.scalar_mean,
            "scalar_std": self.scalar_std,
            "output_mean": self.output_mean,
            "output_std": self.output_std,
        }

    def inverse_transform_inputs_scalar(self, x):
        """Undo normalization for scalar inputs."""
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        return x * self.scalar_std + self.scalar_mean

    def inverse_transform_outputs(self, y):
        y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        return y * self.output_std + self.output_mean

    # ======================================================
    # Reconstruction utilities for cross-checking
    # ======================================================
    def reconstruct_simulation(self, idx, preds=None, fsp=None, vb=None, bin_centers=None):
        """
        Return fully denormalized simulation data for analysis.
        If preds are provided, also reconstruct predicted bins using FSP.
        """
        # ---------------------------------------------
        # Recover input scalars (denormalized)
        # ---------------------------------------------
        X_scalar_norm = self.inputs_scalar[idx].numpy()
        X_scalar = X_scalar_norm * self.scalar_std + self.scalar_mean

        # ---------------------------------------------
        # Recover true target moments (physically meaningful)
        # ---------------------------------------------
        Y_norm = self.targets_moments[idx].numpy()
        Y_true = Y_norm * self.output_std + self.output_mean

        # ---------------------------------------------
        # Extract true distribution & Vtot from dataset
        # ---------------------------------------------
        true_bins = self.true_bins[idx].numpy()  # (sim_len, 57)
        true_vtot = self.true_vtot[idx].numpy()  # (sim_len,)

        result = {
            "sim_index": idx + 1,
            "X_scalar": X_scalar,  # denormalized scalar inputs
            "Y_true": Y_true,  # physical moments
            "true_bins": true_bins,  # true FSP physical distributions
            "true_vtot": true_vtot,  # true Vtot trajectory
        }

        # ---------------------------------------------
        # If predicted moment sequence is provided
        # ---------------------------------------------
        if preds is not None:
            preds = preds.detach().cpu().numpy()

            # denormalize predicted moments
            Y_pred = preds * self.output_std + self.output_mean
            result["Y_pred"] = Y_pred

            # If FSP supplied → reconstruct predicted distribution
            if fsp is not None and vb is not None and bin_centers is not None:
                pred_bins_list = []
                eps = 1e-30

                for t in range(Y_pred.shape[0]):
                    log10_N = Y_pred[t, 0]
                    log10_CMD = Y_pred[t, 1]
                    GSD = Y_pred[t, 2]

                    # convert back to physical scale
                    N_tot = 10 ** log10_N
                    CMD_nm = 10 ** log10_CMD

                    with torch.no_grad():
                        log10_N_t = torch.tensor([log10_N], dtype=torch.float32)
                        log10_CMD_t = torch.tensor([log10_CMD], dtype=torch.float32)
                        GSD_t = torch.tensor([GSD], dtype=torch.float32)

                        bins_t = fsp(
                            log10_Ntot=log10_N_t,
                            log10_CMD_nm=log10_CMD_t,
                            GSD_linear=GSD_t,
                        ).cpu().numpy()[0, 1:]  # remove dummy bin 0

                    pred_bins_list.append(bins_t)

                pred_bins = np.stack(pred_bins_list, axis=0)
                result["pred_bins"] = pred_bins

                # compute predicted Vtot
                pred_bins_torch = torch.tensor(pred_bins).reshape(-1, 1, 57)
                pred_vtot = calc_Vtot_batched(pred_bins_torch, bin_centers).numpy().flatten()
                result["pred_vtot"] = pred_vtot

        return result


