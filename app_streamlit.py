#!/usr/bin/env python3

import os
import sys
import yaml
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import base64

st.set_page_config(layout="wide")


# =============================================================
# BACKGROUND (Optimized for Visibility)
# =============================================================
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Overlay to darken the background slightly for better text contrast */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.4); 
        z-index: -1;
    }}
    </style>
    """, unsafe_allow_html=True)


# Ensure the path matches your local file
set_background("assets/coast_shifted_blended.png")

# =============================================================
# 🔥 UI & INPUT VISIBILITY CONTROL
# =============================================================
st.markdown("""
<style>
/* 1. GLASSMORPHISM EFFECT FOR INPUTS */
/* This creates a semi-transparent box behind inputs so they are readable */
div[data-testid="stNumberInput"], 
div[data-testid="stTextInput"], 
div[data-testid="stSelectbox"],
div[data-testid="stMultiSelect"] {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}

/* 2. TEXT COLOR & FONT SIZES */
/* Ensure all labels are bright white and bold */
label p {
    color: white !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    text-shadow: 1px 1px 2px black;
}

/* 3. INPUT BOX STYLING */
div[data-testid="stNumberInput"] input {
    font-size: 22px !important;
    height: 55px !important;
    background-color: rgba(0, 0, 0, 0.3) !important;
    color: #00FFCC !important; /* Neon cyan for high visibility */
    border: 1px solid #444 !important;
}

/* 4. BUTTON STYLING */
div[data-testid="stButton"] button {
    width: 100%;
    font-size: 24px !important;
    font-weight: bold !important;
    background-color: #007BFF !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 15px !important;
    border: none !important;
    transition: 0.3s;
}

div[data-testid="stButton"] button:hover {
    background-color: #00CCFF !important;
    transform: scale(1.02);
}

/* 5. DATA FRAME VISIBILITY */
[data-testid="stDataFrame"] {
    background-color: rgba(255, 255, 255, 0.9) !important;
    padding: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =============================================================
# PROJECT IMPORTS
# =============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model_moment import MomentPredictor
from fsp_c import FSPReconstructor, vb, bin_centers_np
from CollisionFrequency import CollisionKernal

# =============================================================
# CONSTANTS
# =============================================================
SIM_LEN = 15

THETA_GRID = torch.tensor([
    0.0,0.09434,0.19527,0.28104,0.3949,
    0.49161,0.59367,0.67387,0.7859,0.87389,
    0.99673,1.97112,2.97547,3.98272,4.96214
], dtype=torch.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# LOAD MODEL
# =============================================================
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.yaml")
CHECKPOINT  = os.path.join(SCRIPT_DIR, "cpt_with_split_data",
                           "best_model_mom_bins_vtot_s3007_m1_b0.0001_v0.001_u1_r1_0317_1638.pt")
SCALER_FILE = os.path.join(SCRIPT_DIR, "cpt_with_split_data", "scaler_stats.npz")

config = yaml.safe_load(open(CONFIG_FILE))
scaler_stats = dict(np.load(SCALER_FILE))

@st.cache_resource
def load_model():
    model_cfg = config["model"]

    model = MomentPredictor(
        input_dim_scalar=7,
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        nhead=model_cfg["nhead"],
        max_timesteps=SIM_LEN,
        model_type="transformer",
        output_mean=scaler_stats["output_mean"],
        output_std=scaler_stats["output_std"],
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

model = load_model()
fsp = FSPReconstructor(vb).to(DEVICE)

scalar_mean = torch.tensor(scaler_stats["scalar_mean"], dtype=torch.float32).view(1,1,-1).to(DEVICE)
scalar_std  = torch.tensor(scaler_stats["scalar_std"], dtype=torch.float32).view(1,1,-1).to(DEVICE)

# =============================================================
# UI LAYOUT (RIGHT SIDE)
# =============================================================
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

col_left, col_main = st.columns([1,2])

with col_main:
    st.markdown("<p style='font-size:28px; font-weight:600;'>Initial Conditions (t = 0)</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3,3])

    with col1:
        Temperature = st.number_input("Temperature (K)", 293.0, 303.0, 300.0)
        Pressure = st.number_input("Pressure (Pa)", 101325.0, 101325.0, 101325.0)
        Density_particle = st.number_input("Particle Density (kg/m³)", 500.0, 5000.0, 2800.0)

    with col2:
        Ntot_0 = st.number_input("Initial N_tot (#/m³)", 2e10, 1e16, 1e12, format="%.3e")
        CMD_nm_0 = st.number_input("Initial CMD (nm)", 10.0, 15000.0, 100.0)
        GSD_0 = st.number_input("Initial σg", 1.1, 2.5, 1.5)

# =============================================================
# RUN SIMULATION
# =============================================================
#st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
if st.button("Run Rollout Simulation"):

    CMD_m = CMD_nm_0 * 1e-9

    kernel = CollisionKernal(
        T=Temperature,
        P=Pressure,
        db1=CMD_m,
        db2=CMD_m,
        Rho_particle=Density_particle,
        CMD=CMD_m,
        Nconc_tot_0=Ntot_0
    )

    tau = kernel.Char_coagT()

    st.markdown(
        fr"<p style='font-size:26px; color:#111111;'>"
        fr"Characteristic Coagulation Time (τ) = {tau:.2e} s"
        "</p>",
        unsafe_allow_html=True
    )

    # MODEL INPUT
    x_roll = torch.zeros((1, SIM_LEN, 7)).to(DEVICE)
    x_roll[0,:,0] = Temperature
    x_roll[0,:,1] = Density_particle
    x_roll[0,:,2] = Pressure
    x_roll[0,:,3] = THETA_GRID.to(DEVICE)

    x_roll[0,0,4] = np.log10(Ntot_0)
    x_roll[0,0,5] = np.log10(CMD_nm_0)
    x_roll[0,0,6] = GSD_0

    x_roll = (x_roll - scalar_mean) / scalar_std

    with torch.no_grad():
        y = model(x_roll)
        y = model.moments_to_physical(y)

    logN, logCMD, GSD = y[0,:,0].cpu(), y[0,:,1].cpu(), y[0,:,2].cpu()

    Ntot = 10**logN
    CMD  = 10**logCMD

    # =============================================================
    # PLOTS
    # =============================================================
    fig, ax = plt.subplots(1,3, figsize=(15,4))

    ax[0].plot(THETA_GRID, Ntot)
    ax[0].set_yscale("log")
    ax[0].set_title(r"N$_{tot}$ vs $\theta$", fontsize=20)

    ax[1].plot(THETA_GRID, CMD)
    ax[1].set_title(r"CMD vs $\theta$", fontsize=20)

    ax[2].plot(THETA_GRID, GSD)
    ax[2].set_title(r"$\sigma_{g}$ vs $\theta$", fontsize=20)

    for a in ax:
        a.grid(True)
        a.tick_params(labelsize=16)
        a.set_xlabel(r"$\theta$ = t/$\tau$", fontsize=20)

    st.pyplot(fig)

    # =============================================================
    # TABLE
    # =============================================================
    df = pd.DataFrame({
        "timestep": np.arange(SIM_LEN),
        "θ": THETA_GRID,
        "time (s)": THETA_GRID * tau,
        "N_tot (#/m3)": Ntot,
        "CMD (nm)": CMD,
        "GSD": GSD
    })

    st.subheader("Trajectory Table")
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "results.csv"
    )

    # --------------------------------------------------------
    # PSD Evolution Over All Timesteps
    # --------------------------------------------------------

    logN_tensor = torch.tensor(logN, dtype=torch.float32).to(DEVICE)
    logCMD_tensor = torch.tensor(logCMD, dtype=torch.float32).to(DEVICE)
    GSD_tensor = torch.tensor(GSD, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        bins_all = fsp(
            log10_Ntot=logN_tensor,
            log10_CMD_nm=logCMD_tensor,
            GSD_linear=GSD_tensor
        )

    # Drop dummy bin 0
    bins_all = bins_all[:, 1:].cpu().numpy()  # shape (15, 57)

    diam_nm = bin_centers_np * 1e9

    # --------------------------------------------------------
    # Plot PSD Evolution
    # --------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(7, 5))

    cmap = plt.cm.viridis

    for t in range(SIM_LEN):
        ax3.plot(
            diam_nm,
            bins_all[t],
            color=cmap(t / SIM_LEN),
            alpha=0.9
        )

    ax3.set_xscale("log")
    ax3.set_xlabel("Diameter (nm)", fontsize=20)
    ax3.set_ylabel("Number Concentration", fontsize=20)
    ax3.set_title("PSD Evolution Over Time", fontsize=20)
    ax3.grid(True, which="both")
    ax3.tick_params(axis="x", labelsize=16)
    ax3.tick_params(axis="y", labelsize=16)
    ax3.set_xlim(5, 1e5)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(THETA_GRID.cpu().numpy())
    cbar = plt.colorbar(sm, ax=ax3)
    cbar.set_label(r"$\theta$ = t/$\tau$", fontsize=20)

    st.pyplot(fig3)