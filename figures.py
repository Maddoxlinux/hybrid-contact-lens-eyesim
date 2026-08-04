"""
Generate publication-style figures for the report (no Streamlit needed):

    python figures.py [error_D] [pupil_mm]

Writes PNGs to ./figures/ for the normal / uncorrected / corrected eye:
spot diagrams, MTF and the chromatic-aberration curve.
"""
import sys, os
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eyesim.eye import build_navarro_eye
from eyesim.lens import build_hybrid_lens
from eyesim.metrics import (spot_diagram, geometric_mtf,
                            longitudinal_chromatic_aberration)
from eyesim.optimize import optimize_hybrid_lens
from eyesim.dispersion import LAMBDA_GREEN

ERR = float(sys.argv[1]) if len(sys.argv) > 1 else -3.0
PUP = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
LAMS = np.linspace(0.45, 0.65, 9)
SPOT = [(0.450, "#3b5bff"), (0.550, "#2ca02c"), (0.650, "#e23b3b")]
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

print(f"Building states for error {ERR:+.1f} D, pupil {PUP:.1f} mm ...")
normal = build_navarro_eye(0.0, PUP, LAMBDA_GREEN)
uncorr = build_navarro_eye(ERR, PUP, LAMBDA_GREEN)
builder = partial(build_navarro_eye, ERR, pupil_diameter=PUP, lam_ref_um=LAMBDA_GREEN)
opt = optimize_hybrid_lens(builder, PUP, wavelengths=LAMS[::2])
lens = build_hybrid_lens(opt["refractive_power_D"], opt["diffractive_power_D"],
                         LAMBDA_GREEN)
corrected = build_navarro_eye(ERR, PUP, LAMBDA_GREEN, front_optics=lens.surfaces)
states = [("Normal eye", normal), ("Uncorrected", uncorr), ("Corrected", corrected)]
print(f"  base {opt['refractive_power_D']:.2f} D, diffractive "
      f"{opt['diffractive_power_D']:.2f} D, residual LCA {opt['lca_D']:.2f} D")

# ---- spot diagrams -------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, (title, sysobj) in zip(axes, states):
    for lam, c in SPOT:
        s = spot_diagram(sysobj.system, PUP, lam, rings=14)
        ax.scatter(s["x"], s["y"], s=3, color=c, alpha=0.6,
                   label=f"{lam*1000:.0f} nm")
    ax.set_title(title); ax.set_aspect("equal")
    ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
    ax.set_xlabel("µm"); ax.grid(alpha=0.25)
axes[0].set_ylabel("µm"); axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle(f"Retinal spot diagrams  (error {ERR:+.1f} D, pupil {PUP:.0f} mm)")
fig.tight_layout(); fig.savefig(f"{OUT}/spots.png", dpi=130)
print(f"  wrote {OUT}/spots.png")

# ---- MTF -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for (title, sysobj), c in zip(states, ["#555", "#e23b3b", "#2ca02c"]):
    f, m = geometric_mtf(sysobj.system, PUP, LAMS[::2], nrays=25000, N=224)
    ax.plot(f, m, label=title, color=c, lw=2)
ax.set_xlim(0, 60); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
ax.set_xlabel("Spatial frequency (cycles/degree)"); ax.set_ylabel("MTF")
ax.set_title("Polychromatic MTF (photopic-weighted)")
fig.tight_layout(); fig.savefig(f"{OUT}/mtf.png", dpi=130)
print(f"  wrote {OUT}/mtf.png")

# ---- LCA -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for title, sysobj, c in [("Uncorrected", uncorr, "#e23b3b"),
                         ("Corrected", corrected, "#2ca02c")]:
    d = longitudinal_chromatic_aberration(sysobj.system, LAMS, LAMBDA_GREEN)
    ax.plot(d["wavelengths"] * 1000, d["power_rel_D"], "o-", color=c, lw=2,
            label=f"{title}  (LCA {d['lca_D']:.2f} D)")
ax.axhline(0, color="0.6", lw=0.8); ax.grid(alpha=0.3); ax.legend()
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Relative equivalent power (D), ref 550 nm")
ax.set_title("Longitudinal chromatic aberration")
fig.tight_layout(); fig.savefig(f"{OUT}/lca.png", dpi=130)
print(f"  wrote {OUT}/lca.png")
print("Done.")
