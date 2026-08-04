"""
Pure-Python compute layer shared by the Vercel serverless function.

`simulate(params)` builds the normal / uncorrected / corrected eye states and
returns a JSON-serialisable dict with spot points, MTF and chromatic-aberration
curves plus summary metrics.  Kept free of any web framework so it can be unit
tested directly with NumPy.
"""
from __future__ import annotations
import os, sys
from functools import partial

# make the vendored `eyesim` package importable when this file runs as a
# Vercel function (its own directory is not always on sys.path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from eyesim.eye import build_navarro_eye
from eyesim.lens import build_hybrid_lens, auto_fit_lens
from eyesim.metrics import (spot_diagram, geometric_mtf,
                            longitudinal_chromatic_aberration)
from eyesim.optimize import optimize_hybrid_lens
from eyesim.dispersion import LAMBDA_GREEN

LCA_LAMS = np.linspace(0.45, 0.65, 9)
MTF_LAMS = LCA_LAMS[::2]
SPOT_LAMS = [0.450, 0.550, 0.650]


def _round(a, nd=3):
    return [round(float(v), nd) for v in a]


def _state_payload(eye, pupil):
    spots = {}
    for lam in SPOT_LAMS:
        s = spot_diagram(eye.system, pupil, lam, rings=10)
        pts = np.stack([s["x"], s["y"]], axis=1) if s["x"].size else np.empty((0, 2))
        spots[f"{lam*1000:.0f}"] = [_round(p, 2) for p in pts]
    f, m = geometric_mtf(eye.system, pupil, MTF_LAMS, nrays=8000, N=128)
    lca = longitudinal_chromatic_aberration(eye.system, LCA_LAMS, LAMBDA_GREEN)
    rms = spot_diagram(eye.system, pupil, LAMBDA_GREEN)["rms_um"]
    return dict(
        spots=spots,
        mtf=[[round(float(a), 2), round(float(b), 4)] for a, b in zip(f, m)],
        lca=[[round(float(w * 1000), 1), round(float(p), 4)]
             for w, p in zip(lca["wavelengths"], lca["power_rel_D"])],
        rms_um=round(float(rms), 1),
        lca_D=round(float(lca["lca_D"]), 3),
    )


def simulate(params: dict) -> dict:
    error_D = float(params.get("error_D", -3.0))
    pupil = float(params.get("pupil", 4.0))
    use_lens = bool(params.get("use_lens", True))
    use_diff = bool(params.get("use_diffractive", True))
    material = str(params.get("material", "silicone-hydrogel"))
    design_lam = float(params.get("design_lam_nm", 550)) / 1000.0
    mode = str(params.get("mode", "Auto-fit"))
    man_base = float(params.get("man_base", 35.0))
    man_diff = float(params.get("man_diff", 4.5))

    normal = build_navarro_eye(0.0, pupil, LAMBDA_GREEN)
    uncorr = build_navarro_eye(error_D, pupil, LAMBDA_GREEN)

    lens_info = None
    corrected = uncorr
    if use_lens:
        builder = partial(build_navarro_eye, error_D, pupil_diameter=pupil,
                          lam_ref_um=LAMBDA_GREEN)
        if mode == "Manual":
            base, diff, a4 = man_base, (man_diff if use_diff else 0.0), 0.0
        elif mode == "Optimize":
            res = optimize_hybrid_lens(builder, pupil, design_lam, material,
                                       wavelengths=MTF_LAMS)
            base = res["refractive_power_D"]
            diff = res["diffractive_power_D"] if use_diff else 0.0
            a4 = res["a4"]
        else:  # Auto-fit
            base, diff = auto_fit_lens(builder, pupil, material, design_lam,
                                       use_diff)
            a4 = 0.0
        lens = build_hybrid_lens(base, diff, design_lam, material, a4=a4)
        corrected = build_navarro_eye(error_D, pupil, LAMBDA_GREEN,
                                      front_optics=lens.surfaces)
        lens_info = dict(base=round(base, 2), diff=round(diff, 2),
                         R_front=round(lens.R_front, 2), R_back=round(lens.R_back, 2),
                         design_nm=round(design_lam * 1000, 0))

    return dict(
        params=dict(error_D=error_D, pupil=pupil, use_lens=use_lens,
                    use_diffractive=use_diff, material=material,
                    design_lam_nm=design_lam * 1000, mode=mode),
        lens=lens_info,
        states=dict(
            normal=_state_payload(normal, pupil),
            uncorrected=_state_payload(uncorr, pupil),
            corrected=_state_payload(corrected, pupil),
        ),
    )
