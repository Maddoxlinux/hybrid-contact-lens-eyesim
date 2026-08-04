"""
Numerical optimisation of the hybrid contact lens.

Given an ametropic eye, find the refractive base power, the diffractive add
power and (optionally) the 4th-order diffractive coefficient that minimise a
cost combining:

    * the polychromatic RMS spot radius on the retina  (monochromatic blur +
      chromatic blur), and
    * the residual longitudinal chromatic aberration.

The fitting helpers in `lens.py` give an excellent starting point (paraxial
base power + LCA-nulling diffractive power); this module refines them with a
real ray-traced, photopic-weighted cost.
"""

from __future__ import annotations
import numpy as np

from ._opt import nelder_mead
from .dispersion import PHOTOPIC, LAMBDA_GREEN, LAMBDA_BLUE, LAMBDA_RED
from .lens import build_hybrid_lens, fit_base_power, fit_diffractive_power
from .metrics import spot_diagram, longitudinal_chromatic_aberration


def polychromatic_rms(system, pupil_diameter, wavelengths, weights):
    """Photopic-weighted RMS spot radius (microns), including chromatic blur."""
    # combine all wavelengths about a common (green) centroid so that lateral
    # chromatic and defocus blur are both counted
    xs, ys, ws = [], [], []
    for lam, w in zip(wavelengths, weights):
        s = spot_diagram(system, pupil_diameter, lam, rings=10)
        if s["x"].size:
            xs.append(s["x"] + (s["cx"] * 1000.0))
            ys.append(s["y"] + (s["cy"] * 1000.0))
            ws.append(np.full(s["x"].size, w))
    if not xs:
        return np.inf
    x = np.concatenate(xs); y = np.concatenate(ys); wv = np.concatenate(ws)
    cx = np.average(x, weights=wv); cy = np.average(y, weights=wv)
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return float(np.sqrt(np.average(r2, weights=wv)))


def optimize_hybrid_lens(eye_builder, pupil_diameter,
                         design_lam=LAMBDA_GREEN, material="silicone-hydrogel",
                         wavelengths=None, refine_a4=False,
                         w_chromatic=0.5):
    """Optimise the hybrid lens for a given ametropic eye.

    `eye_builder(front_optics=...)` returns an EyeModel.  Returns a dict with
    the optimal parameters and the resulting metrics.
    """
    if wavelengths is None:
        wavelengths = np.array([0.45, 0.50, 0.55, 0.60, 0.65])
    weights = PHOTOPIC(wavelengths)
    weights = weights / weights.sum()

    # --- physics-based starting point ------------------------------------
    P0 = fit_base_power(eye_builder, pupil_diameter, design_lam, material, design_lam)
    Pd0 = fit_diffractive_power(eye_builder, P0, pupil_diameter, material, design_lam)
    x0 = [P0, Pd0] + ([0.0] if refine_a4 else [])

    def make_system(params):
        Pr, Pd = params[0], params[1]
        a4 = params[2] if refine_a4 else 0.0
        lens = build_hybrid_lens(Pr, Pd, design_lam, material, a4=a4)
        return eye_builder(front_optics=lens.surfaces), lens

    def cost(params):
        eye, _ = make_system(params)
        rms = polychromatic_rms(eye.system, pupil_diameter, wavelengths, weights)
        lca = abs(longitudinal_chromatic_aberration(
            eye.system, wavelengths, design_lam)["lca_D"])
        return rms + w_chromatic * 1000.0 * lca * 0.02   # scale LCA into microns-ish

    params, _ = nelder_mead(cost, x0, step=0.5,
                            xatol=1e-3, fatol=1e-2, maxiter=400)
    eye, lens = make_system(params)
    lca = longitudinal_chromatic_aberration(eye.system, wavelengths, design_lam)
    rms = polychromatic_rms(eye.system, pupil_diameter, wavelengths, weights)
    return dict(
        refractive_power_D=float(params[0]),
        diffractive_power_D=float(params[1]),
        a4=float(params[2]) if refine_a4 else 0.0,
        start=dict(refractive_power_D=P0, diffractive_power_D=Pd0),
        lca_D=lca["lca_D"], poly_rms_um=rms,
        lens=lens, eye=eye,
    )
