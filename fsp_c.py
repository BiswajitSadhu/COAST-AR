import numpy as np
import math
import matplotlib.pyplot as plt
import torch
# torch.set_default_dtype(torch.float64)

class FSPReconstructor_old_code(torch.nn.Module):
    def __init__(self, vb, ac=0.147, eps=1e-30):
        super().__init__()
        self.ac = ac
        self.eps = eps
        self.register_buffer("vb", torch.tensor(vb, dtype=torch.float32))  # float64

    def forward(self, log10_Ntot, log10_CMD_nm, GSD_linear):
        """
        log10_Ntot   : log10 of total number concentration
        log10_CMD_nm : log10 of CMD in nm
        GSD_linear   : geometric standard deviation (linear > 1), NOT log10!
        """
        log10_Ntot   = log10_Ntot.view(-1, 1)
        log10_CMD_nm = log10_CMD_nm.view(-1, 1)
        GSD_linear   = GSD_linear.view(-1, 1)

        # Undo log10 for Ntot and CMD (as before)
        Ntot = torch.clamp(10 ** log10_Ntot, min=self.eps)
        CMD_nm = torch.clamp(10 ** log10_CMD_nm, min=self.eps)
        CMD_m = CMD_nm * 1e-9

        # HERE is the key change: GSD is already linear
        GSD = torch.clamp(GSD_linear, min=1.0 + 1e-6)

        CMDV = (math.pi / 6.0) * CMD_m**3

        vb = self.vb
        B = Ntot.shape[0]
        Nb = len(vb)

        vb_i   = vb[:-1].view(1, -1).expand(B, -1)
        vb_ip1 = vb[1:].view(1, -1).expand(B, -1)

        lnG = torch.log(GSD)
        sqrt18 = math.sqrt(18.0)

        xx = torch.log(vb_ip1 / CMDV) / (sqrt18 * lnG)
        yy = torch.log(vb_i   / CMDV) / (sqrt18 * lnG)
        xx = torch.clamp(xx, min=-50.0, max=50.0)
        yy = torch.clamp(yy, min=-50.0, max=50.0)

        ac = self.ac
        pi = math.pi

        xex = torch.exp(-xx*xx*((4/pi) + ac*xx*xx) / (1 + ac*xx*xx))
        yex = torch.exp(-yy*yy*((4/pi) + ac*yy*yy) / (1 + ac*yy*yy))

        term1 = xx * torch.sqrt(torch.clamp(1 - xex, min=0.0)) / (torch.abs(xx) + 1e-30)
        term2 = yy * torch.sqrt(torch.clamp(1 - yex, min=0.0)) / (torch.abs(yy) + 1e-30)

        term1 = torch.where(torch.abs(xx) < 1e-08, torch.zeros_like(term1), term1)
        term2 = torch.where(torch.abs(yy) < 1e-08, torch.zeros_like(term2), term2)

        Nint = 0.5 * (term1 - term2) * Ntot

        N = torch.zeros((B, Nb), dtype=torch.float32, device=log10_Ntot.device)
        N[:, 1:] = Nint
        return N



class FSPReconstructor(torch.nn.Module):
    def __init__(self, vb, ac=0.147, eps=1e-30):
        super().__init__()
        self.ac = ac
        self.eps = eps

        vb = torch.tensor(vb, dtype=torch.float64)
        self.register_buffer("vb", vb)
        self.register_buffer("log_vb", torch.log(vb))

    def forward(self, log10_Ntot, log10_CMD_nm, GSD_linear):

        log10_Ntot   = log10_Ntot.view(-1,1)
        log10_CMD_nm = log10_CMD_nm.view(-1,1)
        GSD_linear   = GSD_linear.view(-1,1)

        Ntot = torch.clamp(10**log10_Ntot, min=self.eps).double()
        CMD_nm = torch.clamp(10**log10_CMD_nm, min=self.eps).double()
        CMD_m = CMD_nm * 1e-9

        GSD = torch.clamp(GSD_linear, min=1.000001).double()

        CMDV = (math.pi/6.0)*CMD_m**3
        log_CMDV = torch.log(CMDV)

        vb = self.vb
        log_vb = self.log_vb

        B = Ntot.shape[0]
        Nb = len(vb)

        log_vb_i = log_vb[:-1].view(1,-1).expand(B,-1)
        log_vb_ip1 = log_vb[1:].view(1,-1).expand(B,-1)

        lnG = torch.log(GSD)
        lnG = torch.clamp(lnG, min=1e-3)

        sqrt18 = math.sqrt(18.0)

        xx = (log_vb_ip1 - log_CMDV) / (sqrt18*lnG)
        yy = (log_vb_i   - log_CMDV) / (sqrt18*lnG)

        ac = self.ac
        pi = math.pi

        argx = -xx*xx*((4/pi) + ac*xx*xx)/(1+ac*xx*xx)
        argy = -yy*yy*((4/pi) + ac*yy*yy)/(1+ac*yy*yy)

        argx = torch.clamp(argx, min=-60)
        argy = torch.clamp(argy, min=-60)

        xex = torch.exp(argx)
        yex = torch.exp(argy)

        term1 = torch.sign(xx)*torch.sqrt(torch.clamp(1-xex,min=0))
        term2 = torch.sign(yy)*torch.sqrt(torch.clamp(1-yex,min=0))

        Nint = 0.5*(term1-term2)*Ntot

        N = torch.zeros((B,Nb),dtype=torch.float64,device=log10_Ntot.device)
        N[:,1:] = Nint

        return N.float()
# -------------------------------------------------------------
#  BIN GENERATION FROM YOUR FIXED BIN CENTERS
# -------------------------------------------------------------
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

# compute geometric bin edges
geom_mid = np.sqrt(bin_centers_np[:-1] * bin_centers_np[1:])
edges = np.zeros(len(bin_centers_np) + 1)
edges[1:-1] = geom_mid
edges[0] = bin_centers_np[0]**2 / geom_mid[0]
edges[-1] = bin_centers_np[-1]**2 / geom_mid[-1]

# compute volume bins
vb = (math.pi/6.0) * (edges**3)

