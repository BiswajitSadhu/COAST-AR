#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import yaml
from contextlib import nullcontext
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from sklearn.model_selection import GroupShuffleSplit
from utils import debug_print_bins_and_moments
from preprocess_moment import AerosolSimDataset
from model_moment import MomentPredictor, PhysicsLoss
from utils import bin_centers, calc_Vtot_batched


# ================================================================
# Load Config
# ================================================================
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()

# ================================================================
# Reproducibility
# ================================================================
SEED = config.get("seed", 42)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================================================================
# Device Setup
# ================================================================
DEVICE = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ===============================================================
# Autograd Anomaly Detection (for debugging NaNs)
# ================================================================
use_debug_gradient = False
if use_debug_gradient:
    torch.autograd.set_detect_anomaly(True)
    print("⚠️ Autograd anomaly detection ENABLED")
# ===============================================================

# ================================================================
# Directory Setup
# ================================================================
os.makedirs(config["checkpoint_dir"], exist_ok=True)
os.makedirs(config["log_dir"], exist_ok=True)
#os.makedirs(config.get("split_dir", "splits"), exist_ok=True)

# ================================================================
# Dataset
#   - Uses ONLY scalar inputs
#   - Targets are normalized moments (3)
#   - Returns true FSP bins + Vtot for physics loss
# ================================================================
dataset = AerosolSimDataset(
    csv_file=config["data_csv"],
    sim_len=config["sim_len"],  # should be 15 with your new dataset
    fit_scaler=True,
    scaler_stats=None,
    log_transform=True,  # log10 on N_tot & CMD_nm inside dataset
)

print("\n--- Dataset Overview ---")
print(f"Total simulations: {len(dataset)}")
print(f"Scalar input dim: {dataset.inputs_scalar.shape[-1]}")
print(f"Moments output dim: {dataset.targets_moments.shape[-1]}")  # should be 3
print(f"Timesteps per sim: {dataset.sim_len}")
print("====================================================\n")

# ================================================================
# Save Scaler Stats for Evaluation
# ================================================================
scaler_stats = dataset.get_scaler_stats()
scaler_path = os.path.join(config["checkpoint_dir"], "scaler_stats.npz")
np.savez(scaler_path, **scaler_stats)
print(f"💾 Saved scaler statistics to: {scaler_path}")

# ================================================================
# Dataset Split (group-wise by simulation)
# ================================================================
# ================================================================
# Dataset Split (USE PRE-SAVED STRATIFIED SPLITS — NO REGENERATION)
# ================================================================
saved_split_dir = "DATASET/GROUPKFOLD_STRATIFIED/saved_splits"
split_meta_path = os.path.join(saved_split_dir, "split_metadata.npz")

if not os.path.exists(split_meta_path):
    raise FileNotFoundError(
        f"❌ split_metadata.npz not found in {saved_split_dir}. "
        "splits are already created — please check the path."
    )

print(f"📂 Loading pre-saved stratified splits from: {saved_split_dir}")

# Load simulation-level splits (IMPORTANT: these are sim_index, not row indices)
meta = np.load(split_meta_path, allow_pickle=True)
train_sims = meta["train_sims"]
val_sims   = meta["val_sims"]
test_sims  = meta["test_sims"]

# Convert simulation splits → dataset indices (keeps all 15 timesteps per simulation)
all_sim_indices = np.array(dataset.sim_index_values)

train_idx = np.where(np.isin(all_sim_indices, train_sims))[0]
val_idx   = np.where(np.isin(all_sim_indices, val_sims))[0]
test_idx  = np.where(np.isin(all_sim_indices, test_sims))[0]

# Safety checks (VERY IMPORTANT)
assert len(set(train_sims) & set(val_sims)) == 0, "Overlap between train and val simulations!"
assert len(set(train_sims) & set(test_sims)) == 0, "Overlap between train and test simulations!"
assert len(set(val_sims) & set(test_sims)) == 0, "Overlap between val and test simulations!"

print("\n📊 Using FIXED Physics-Aware Stratified Splits (t=0 based)")
print(f"Train samples: {len(train_idx)}")
print(f"Val samples:   {len(val_idx)}")
print(f"Test samples:  {len(test_idx)}")

print(f"Train simulations: {len(np.unique(train_sims))}")
print(f"Val simulations:   {len(np.unique(val_sims))}")
print(f"Test simulations:  {len(np.unique(test_sims))}")
print("====================================================\n")

# ================================================================
# Subsets and Loaders
# ================================================================
train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=config["batch_size"],
    shuffle=True,
    drop_last=False,
)
val_loader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=config["batch_size"],
    shuffle=False,
    drop_last=False,
)
test_loader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=config["batch_size"],
    shuffle=False,
    drop_last=False,
)

print(f"\n📊 Split Summary:")
print(f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
print("====================================================\n")

# ================================================================
# Model Setup
#   - MomentPredictor: scalar sequence → normalized moments
# ================================================================
num_scalar = dataset.inputs_scalar.shape[-1]  # 7
num_outputs = dataset.targets_moments.shape[-1]  # 3
sim_len = dataset.sim_len

model_cfg = config["model"]

model = MomentPredictor(
    input_dim_scalar=num_scalar,
    hidden_dim=model_cfg["hidden_dim"],
    num_layers=model_cfg["num_layers"],
    dropout=model_cfg["dropout"],
    nhead=model_cfg["nhead"],
    max_timesteps=sim_len,
    model_type=model_cfg.get("type", "transformer"),
    output_mean=scaler_stats["output_mean"],
    output_std=scaler_stats["output_std"],
).to(DEVICE)


# Weight initialization (keep this)
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


model.apply(init_weights)

print(model)
print("====================================================\n")

# ================================================================
# Loss Function: PhysicsLoss with Kendall weighting
# ================================================================
loss_cfg = config.get("loss_fn", {})
use_kendall = loss_cfg.get("use_uncertainty", True)

loss_fn = PhysicsLoss(
    n_bins=57,
    output_mean=scaler_stats["output_mean"],
    output_std=scaler_stats["output_std"],
    fsp=model.fsp,
    mom_weight=loss_cfg["mom_weight"],
    bin_weight=loss_cfg["bin_weight"],
    vtot_weight=loss_cfg["vtot_weight"],
    use_uncertainty=loss_cfg["use_uncertainty"],
).to(DEVICE)


# ================================================================
# Optimizer & Scheduler
#   - Optimizer includes model + loss (for Kendall log_sigmas)
# ================================================================
optimizer_cfg = config["optimizer"]
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=optimizer_cfg["lr"],
    weight_decay=optimizer_cfg["weight_decay"],
)

scheduler_cfg = config["scheduler"]
scheduler = ReduceLROnPlateau(
    optimizer,
    mode=scheduler_cfg["mode"],
    factor=scheduler_cfg["factor"],
    patience=scheduler_cfg["patience"],
)

# ================================================================
# TensorBoard
# ================================================================
writer = SummaryWriter(config["log_dir"])

# ================================================================
# Training Loop
# ================================================================
best_val_loss = float("inf")
patience_counter = 0
global_step = 0
max_epochs = config["epochs"]
print_freq = config.get("print_frequency", 20)
debug = config["loss_fn"].get("debug")
print('debug:', debug)
for epoch in range(max_epochs):
    model.train()
    train_loss_accum = 0.0

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    print(f"\nEpoch {epoch + 1}/{max_epochs} | LR: {current_lr:.3e}")

    for batch_idx, (x_scalar, y_mom_norm, true_bins, true_vtot) in enumerate(train_loader):

        x_scalar   = x_scalar.to(DEVICE)      # (B,T,scalar)
        y_mom_norm = y_mom_norm.to(DEVICE)    # (B,T,3)
        true_bins  = true_bins.to(DEVICE)     # (B,T,57)
        true_vtot  = true_vtot.to(DEVICE)     # (B,T)

        optimizer.zero_grad()

        # ---------------------------------------------------
        # 1) Model → predict normalized moments
        # ---------------------------------------------------
        pred_mom_norm = model(x_scalar)       # (B,T,3)

        #pred_mom_phys = model.moments_to_physical(pred_mom_norm)
        #pred_vtot = calc_Vtot_batched(pred_bins, bin_centers)

        # ---------------------------------------------------
        # 2) Compute loss (loss_fn handles: inverse_norm, clamping, FSP)
        # ---------------------------------------------------
        if debug:
            total_loss, details, pred_bins, pred_mom_phys = loss_fn(
                pred_mom_norm=pred_mom_norm,
                true_mom_norm=y_mom_norm,
                true_bins=true_bins,
                true_vtot=true_vtot,
                debug=True,
                epoch=epoch
            )
            pred_vtot = calc_Vtot_batched(pred_bins, bin_centers)


            debug_print_bins_and_moments(
                pred_bins=pred_bins.to(loss_fn.bin_centers.dtype),
                true_bins=true_bins.to(loss_fn.bin_centers.dtype),
                pred_mom_phys=pred_mom_phys,
                true_mom_phys=loss_fn.moments_to_physical(y_mom_norm),
                pred_vtot=pred_vtot,
                true_vtot=true_vtot,
                batch_idx=batch_idx,
            )

        else:
            total_loss, details = loss_fn(
                pred_mom_norm=pred_mom_norm,
                true_mom_norm=y_mom_norm,
                true_bins=true_bins,
                true_vtot=true_vtot,
                debug=False,
                epoch=epoch
            )

        #print('pred_mom_norm', pred_mom_norm, total_loss, details)

        # ---------------------------------------------------
        # 3) Backprop
        # ---------------------------------------------------
        #total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(
        #    list(model.parameters()) + list(loss_fn.parameters()), 1.0
        #)
        #optimizer.step()
        
        # ---------------------------------------------------
        # 3) Backprop (with anomaly detection)
        # ---------------------------------------------------
        ctx = torch.autograd.detect_anomaly() if use_debug_gradient else nullcontext()

        try:
            with ctx:
                total_loss.backward()
        except RuntimeError as e:
            print("\n🚨 AUTOGRAD ANOMALY DETECTED 🚨")
            print(e)

            print("\nBatch:", batch_idx)
            print("Epoch:", epoch)

            print("\nLoss values:")
            print(details)

            print("\nPredicted moment stats:")
            pred_phys = model.moments_to_physical(pred_mom_norm)
            print("logN:", pred_phys[..., 0].min().item(), pred_phys[..., 0].max().item())
            print("logCMD:", pred_phys[..., 1].min().item(), pred_phys[..., 1].max().item())
            print("GSD:", pred_phys[..., 2].min().item(), pred_phys[..., 2].max().item())

            print("\nPredicted bin stats:")
            print("min:", pred_bins.min().item())
            print("max:", pred_bins.max().item())

            # Save failing batch for reproduction
            torch.save({
                "x_scalar": x_scalar.detach().cpu(),
                "true_bins": true_bins.detach().cpu(),
                "pred_mom_norm": pred_mom_norm.detach().cpu(),
                "pred_bins": pred_bins.detach().cpu(),
                "epoch": epoch,
                "batch": batch_idx
            }, "nan_failure_batch.pt")

            raise e
            
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_fn.parameters()), 1.0
        )
        optimizer.step()

        train_loss_accum += total_loss.item()
        global_step += 1

        # ---------------------------------------------------
        # 4) Print diagnostics
        # ---------------------------------------------------
        if (batch_idx + 1) % print_freq == 0:
            print(
                f"[Train Batch {batch_idx+1:03d}] "
                f"mom={details['loss_mom'].item():.4e} "
                f"bins={details['loss_bins'].item():.4e} "
                f"vtot={details['loss_vtot'].item():.4e} "
                f"total={details['total'].item():.4e}"
            )

    avg_train_loss = train_loss_accum / max(1, len(train_loader))

    # --------------------------
    # Validation
    # --------------------------
    model.eval()
    val_loss_accum = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            x_scalar, y_mom_norm, true_bins, true_vtot = batch
            x_scalar = x_scalar.to(DEVICE)
            y_mom_norm = y_mom_norm.to(DEVICE)
            true_bins = true_bins.to(DEVICE)
            true_vtot = true_vtot.to(DEVICE)

            pred_mom_norm = model(x_scalar)

            pred_mom_phys = model.moments_to_physical(pred_mom_norm)


            logN = pred_mom_phys[..., 0].reshape(-1)
            logCMD = pred_mom_phys[..., 1].reshape(-1)
            GSD = pred_mom_phys[..., 2].reshape(-1)

            bins_all = model.fsp(
                log10_Ntot=logN,
                log10_CMD_nm=logCMD,
                GSD_linear=GSD,
            )
            bins_all = bins_all[:, 1:]
            pred_bins = bins_all.reshape(true_bins.shape)

            if debug:
                val_loss, v_details, _, _ = loss_fn(
                    pred_mom_norm=pred_mom_norm,
                    true_mom_norm=y_mom_norm,
                    true_bins=true_bins,
                    true_vtot=true_vtot,
                    debug=True,
                    epoch=epoch
                )
            else:
                val_loss, v_details = loss_fn(
                    pred_mom_norm=pred_mom_norm,
                    true_mom_norm=y_mom_norm,
                    true_bins=true_bins,
                    true_vtot=true_vtot,
                    debug=True,
                    epoch=epoch
                )

            val_loss_accum += val_loss.item()

    avg_val_loss = val_loss_accum / max(1, len(val_loader))
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch + 1}: Train={avg_train_loss:.4e}, Val={avg_val_loss:.4e}")

    # --------------------------
    # Logging
    # --------------------------
    writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
    writer.add_scalar("Loss/Val_Total", avg_val_loss, epoch)

    if hasattr(loss_fn, "log_sigma_mom"):
        writer.add_scalar("Sigma/log_sigma_mom", loss_fn.log_sigma_mom.item(), epoch)
    if hasattr(loss_fn, "log_sigma_bins"):
        writer.add_scalar("Sigma/log_sigma_bins", loss_fn.log_sigma_bins.item(), epoch)
    if hasattr(loss_fn, "log_sigma_vtot"):
        writer.add_scalar("Sigma/log_sigma_vtot", loss_fn.log_sigma_vtot.item(), epoch)

    # --------------------------
    # Checkpointing
    # --------------------------
    ckpt_path = os.path.join(config["checkpoint_dir"], f"model_epoch{epoch + 1}.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "loss_state_dict": loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
        },
        ckpt_path,
    )

    # Save best model (early stopping)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_path = os.path.join(config["checkpoint_dir"], "best_model.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss_state_dict": loss_fn.state_dict(),
                "val_loss": avg_val_loss,
            },
            best_path,
        )
        print(f"⭐ New best model saved to {best_path}")
    else:
        patience_counter += 1
        if patience_counter >= config["patience"]:
            print(f"⏹️ Early stopping at epoch {epoch + 1}")
            break

writer.close()
print("✅ Training complete.")
