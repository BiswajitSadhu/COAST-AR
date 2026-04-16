import pandas as pd
import torch
import numpy as np
import math

# ================================================================
# Fixed bin centers (meters)
# ================================================================
bin_centers_np = np.array([
    1.5874010519682E-10, 2.0E-10, 2.51984209978975E-10, 3.1748021039364E-10, 4.0E-10,
    5.0396841995795E-10, 6.34960420787281E-10, 8.00000000000001E-10, 1.0079368399159E-09,
    1.26992084157456E-09, 1.6E-09, 2.0158736798318E-09, 2.53984168314912E-09, 3.2E-09,
    4.0317473596636E-09, 5.07968336629824E-09, 6.4E-09, 8.0634947193272E-09, 1.01593667325965E-08,
    1.28E-08, 1.61269894386544E-08, 2.0318733465193E-08, 2.56E-08, 3.22539788773088E-08,
    4.06374669303859E-08, 5.12E-08, 6.45079577546176E-08, 8.12749338607719E-08, 1.024E-07,
    1.29015915509235E-07, 1.62549867721544E-07, 2.048E-07, 2.5803183101847E-07, 3.25099735443088E-07,
    4.096E-07, 5.16063662036941E-07, 6.50199470886175E-07, 8.192E-07, 1.03212732407388E-06,
    1.30039894177235E-06, 1.6384E-06, 2.06425464814776E-06, 2.6007978835447E-06, 3.2768E-06,
    4.12850929629552E-06, 5.2015957670894E-06, 6.5536E-06, 8.25701859259105E-06, 1.04031915341788E-05,
    1.31072E-05, 1.65140371851821E-05, 2.08063830683576E-05, 2.62144E-05, 3.30280743703642E-05,
    4.16127661367152E-05, 5.24288E-05, 6.60561487407284E-05
])

bin_centers = torch.tensor(bin_centers_np, dtype=torch.float32)

# ================================================================
# Earlier CMD/GSD function (USES RATIO/PDF)
# ================================================================
def calc_CMD_GSD_batched(nconc_phys: torch.Tensor, bin_centers: torch.Tensor, eps=1e-30):

    device = nconc_phys.device
    B, T, N = nconc_phys.shape

    # Normalize to PDF
    pdf = nconc_phys / (nconc_phys.sum(-1, keepdim=True) + eps)

    pdf_flat = pdf.reshape(B*T, N)
    cdf_flat = torch.cumsum(pdf_flat, dim=-1)

    percentiles = torch.tensor([0.16, 0.50, 0.84], device=device)
    percentiles_expand = percentiles.view(1,3).expand(B*T,3)

    idx = torch.searchsorted(cdf_flat, percentiles_expand)
    idx = torch.clamp(idx, 1, N-1)

    log_bins = torch.log(bin_centers)
    log_bins_flat = log_bins.expand(B*T, N)

    c0 = torch.gather(cdf_flat, 1, idx-1)
    c1 = torch.gather(cdf_flat, 1, idx)

    y0 = torch.gather(log_bins_flat, 1, idx-1)
    y1 = torch.gather(log_bins_flat, 1, idx)

    slopes = (y1-y0) / (c1-c0 + eps)
    logs = y0 + slopes * (percentiles_expand - c0)

    D16 = torch.exp(logs[:,0])
    D50 = torch.exp(logs[:,1])
    D84 = torch.exp(logs[:,2])

    CMD = D50.reshape(B,T)
    GSD = torch.sqrt(D84 / (D16 + eps)).reshape(B,T)

    return CMD, GSD


# ================================================================
# Earlier Vtot function (USES ABSOLUTE Nconc)
# ================================================================
def calc_Vtot_batched(nconc_phys: torch.Tensor, bin_centers, eps=1e-30):
    device = nconc_phys.device

    nconc_phys = torch.clamp(nconc_phys, min=eps)

    radii = bin_centers / 2
    V_bin = (4.0/3.0) * np.pi * (radii ** 3)
    V_bin = V_bin.view(1,1,-1)            # (1,1,57)
    V_bin = V_bin.to(device)
    Vtot = torch.sum(nconc_phys * V_bin, dim=-1)
    return Vtot


def vtot_from_moments(logN, logCMD, GSD):
    N = 10**logN
    CMD = 10**logCMD

    Vtot = (np.pi/6) * N * CMD**3 * torch.exp(4.5 * torch.log(GSD)**2)
    return Vtot

# utils.py

import numpy as np
import torch

def debug_print_bins_and_moments(
    pred_bins, true_bins,
    pred_mom_phys, true_mom_phys,
    pred_vtot, true_vtot,
    batch_idx
):
    """
    Debug print for Batch 0 only.
    Shows:
        - True vs Pred CMD (nm), GSD, Ntot
        - True vs Pred Vtot
        - Per-bin values, ratios, sums, min/max
    """
    if batch_idx != 0:
        return  # only print for batch 0

    # ============================================================
    # 1) Extract first simulation, timestep 0
    # ============================================================
    pb = pred_bins[0, 0].detach().cpu().numpy()
    tb = true_bins[0, 0].detach().cpu().numpy()

    pred_m = pred_mom_phys[0, 0].detach().cpu().numpy()
    true_m = true_mom_phys[0, 0].detach().cpu().numpy()

    pred_v = float(pred_vtot[0, 0].detach().cpu())
    true_v = float(true_vtot[0, 0].detach().cpu())

    # Physical moments: [log10Ntot, log10CMD_nm, GSD]
    pred_Ntot = 10 ** pred_m[0]
    true_Ntot = 10 ** true_m[0]

    pred_CMD = 10 ** pred_m[1]
    true_CMD = 10 ** true_m[1]

    pred_GSD = pred_m[2]
    true_GSD = true_m[2]

    # ============================================================
    # HEADER
    # ============================================================
    print("\n================ DEBUG: MOMENTS + BINS (batch 0) ==================\n")

    # ============================================================
    # 2) Print Moments Comparison
    # ============================================================
    print(">>> Physical Moments (timestep 0)")
    print(f"{'Quantity':15s} | {'True':>12s} | {'Predicted':>12s} | {'Ratio':>12s}")
    print("-"*55)
    print(f"{'Ntot':15s} | {true_Ntot:12.4e} | {pred_Ntot:12.4e} | {pred_Ntot/true_Ntot:12.4e}")
    print(f"{'CMD (nm)':15s} | {true_CMD:12.4e} | {pred_CMD:12.4e} | {pred_CMD/true_CMD:12.4e}")
    print(f"{'GSD':15s} | {true_GSD:12.4e} | {pred_GSD:12.4e} | {(pred_GSD/true_GSD):12.4e}")

    print("\n>>> Vtot Check")
    print(f"{'Vtot':15s} | {true_v:12.4e} | {pred_v:12.4e} | {pred_v/true_v:12.4e}")

    # ============================================================
    # 3) Per-bin distribution comparison
    print("\n>>> Bin-wise Comparison")
    print("Bin  |       True        Pred        Ratio(pred/true)")
    for i in range(len(pb)):
        true_v_bin = float(tb[i])
        pred_v_bin = float(pb[i])
        ratio = pred_v_bin / true_v_bin if true_v_bin > 0 else float('inf')
        print(f"{i:02d}   {true_v_bin:10.3e}  {pred_v_bin:10.3e}   {ratio:10.3e}")

    # ============================================================
    # 4) Ntot computed from bins
    true_sum = tb.sum()
    pred_sum = pb.sum()

    print("\n>>> Ntot from integrating distribution:")
    print(f"Sum(True bins): {true_sum:.4e}")
    print(f"Sum(Pred bins): {pred_sum:.4e}")
    print(f"Ratio:          {pred_sum/true_sum:.4e}")

    # ============================================================
    # 5) Sanity Checks
    print("\n>>> Min/Max Values")
    print(f"True bins  min={tb.min():.3e}, max={tb.max():.3e}")
    print(f"Pred bins  min={pb.min():.3e}, max={pb.max():.3e}")

    print("=================================================================\n")


# ================================================================
# Ablation Metrics
# ================================================================
def compute_ablation_metrics(pred_bins, true_bins,
                             pred_vtot, true_vtot,
                             pred_mom_phys, true_mom_phys):

    eps = 1e-12

    mae_bins = torch.mean(torch.abs(pred_bins - true_bins)).item()

    rmse_logbins = torch.sqrt(
        torch.mean(
            (torch.log10(pred_bins + eps) -
             torch.log10(true_bins + eps)) ** 2
        )
    ).item()

    vtot_rel_error = torch.mean(
        torch.abs(pred_vtot - true_vtot) /
        (true_vtot + eps)
    ).item()

    rmse_logN = torch.sqrt(
        torch.mean((pred_mom_phys[..., 0] -
                    true_mom_phys[..., 0]) ** 2)
    ).item()

    rmse_logCMD = torch.sqrt(
        torch.mean((pred_mom_phys[..., 1] -
                    true_mom_phys[..., 1]) ** 2)
    ).item()

    rmse_GSD = torch.sqrt(
        torch.mean((pred_mom_phys[..., 2] -
                    true_mom_phys[..., 2]) ** 2)
    ).item()

    p = pred_bins / (pred_bins.sum(dim=-1, keepdim=True) + eps)
    q = true_bins / (true_bins.sum(dim=-1, keepdim=True) + eps)

    m = 0.5 * (p + q)

    jsd = 0.5 * (
        torch.sum(p * torch.log((p + eps) / (m + eps)), dim=-1)
        + torch.sum(q * torch.log((q + eps) / (m + eps)), dim=-1)
    )

    jsd = torch.mean(jsd).item()

    return {
        "mae_bins": mae_bins,
        "rmse_logbins": rmse_logbins,
        "vtot_rel_error": vtot_rel_error,
        "rmse_logN": rmse_logN,
        "rmse_logCMD": rmse_logCMD,
        "rmse_GSD": rmse_GSD,
        "jsd": jsd,
    }

