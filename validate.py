"""
Sanity checks for the eyesim engine.  Run:  python validate.py

Verifies the physics the project depends on:
  1. the nominal Navarro eye is emmetropic (~60 D, sharp on-axis focus);
  2. axial ametropia produces the expected sign and magnitude of blur;
  3. the bare eye shows ~2 D of longitudinal chromatic aberration;
  4. a hybrid refractive-diffractive contact lens both refocuses the eye
     and reduces its chromatic aberration.
"""
import numpy as np
from functools import partial

from eyesim.eye import build_navarro_eye, NOMINAL_AXIAL_LENGTH
from eyesim.raytrace import System, paraxial_power
from eyesim.metrics import (spot_diagram, longitudinal_chromatic_aberration,
                            geometric_mtf)
from eyesim.lens import build_hybrid_lens, fit_base_power, fit_diffractive_power
from eyesim.optimize import optimize_hybrid_lens

LAMS = np.array([0.45, 0.50, 0.55, 0.60, 0.65])
GREEN = 0.55


def line(msg): print(msg)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


line("\n1. Nominal (emmetropic) eye")
eye0 = build_navarro_eye(0.0, pupil_diameter=4.0, lam_ref_um=GREEN)
P, bfd = paraxial_power(System(eye0.system.surfaces[:-1],
                               eye0.system.material_before), GREEN)
s0 = spot_diagram(eye0.system, 4.0, GREEN)
line(f"   equivalent power   = {P:6.2f} D")
line(f"   axial length       = {eye0.axial_length:6.2f} mm")
line(f"   on-axis RMS spot   = {s0['rms_um']:6.2f} um")
check("power in 58-62 D", 58 < P < 62)
check("axial length 22-26 mm", 22 < eye0.axial_length < 26)
# ~10 um on-axis at a 4 mm pupil is the Navarro eye's real spherical
# aberration (diffraction limit alone is ~2.8 um); anything under ~15 um
# means the eye is aberration-limited, not defocused.
check("emmetropic aberration-limited (RMS < 15 um)", s0["rms_um"] < 15)

line("\n2. Ametropia sign & magnitude (pupil 4 mm, green)")
for E in (-3.0, +3.0):
    eye = build_navarro_eye(E, pupil_diameter=4.0, lam_ref_um=GREEN)
    s = spot_diagram(eye.system, 4.0, GREEN)
    dL = eye.axial_length - eye0.axial_length
    line(f"   E={E:+.1f} D  axial {eye.axial_length:5.2f} mm "
         f"(dL={dL:+.2f})  RMS={s['rms_um']:6.1f} um")
    check(f"myopia longer / hyperopia shorter (E={E:+.0f})",
          (E < 0 and dL > 0) or (E > 0 and dL < 0))
    check(f"defocus blur grows (E={E:+.0f})", s["rms_um"] > 20)

line("\n3. Longitudinal chromatic aberration of the bare eye")
lca = longitudinal_chromatic_aberration(eye0.system, LAMS, GREEN)
line("   wavelength  power(D)  rel-power(D)")
for lam, p, pr in zip(lca["wavelengths"], lca["power_D"], lca["power_rel_D"]):
    line(f"     {lam*1000:4.0f} nm  {p:7.2f}  {pr:+6.3f}")
line(f"   LCA (blue-red)     = {lca['lca_D']:5.2f} D")
check("ocular LCA ~1.5-2.5 D", 1.3 < lca["lca_D"] < 2.6)
check("blue focuses stronger than red", lca["power_D"][0] > lca["power_D"][-1])

line("\n4. Hybrid contact-lens correction of a -3 D myope")
builder = partial(build_navarro_eye, -3.0, pupil_diameter=4.0, lam_ref_um=GREEN)
opt = optimize_hybrid_lens(builder, 4.0, design_lam=GREEN, wavelengths=LAMS)
line(f"   base refractive power = {opt['refractive_power_D']:6.2f} D")
line(f"   diffractive add power = {opt['diffractive_power_D']:6.2f} D")
line(f"   residual LCA          = {opt['lca_D']:5.2f} D")
line(f"   poly RMS (corrected)  = {opt['poly_rms_um']:6.2f} um")

uncorr = build_navarro_eye(-3.0, pupil_diameter=4.0, lam_ref_um=GREEN)
s_un = spot_diagram(uncorr.system, 4.0, GREEN)
lca_un = longitudinal_chromatic_aberration(uncorr.system, LAMS, GREEN)
line(f"   (uncorrected green RMS = {s_un['rms_um']:6.1f} um, "
     f"LCA = {lca_un['lca_D']:4.2f} D)")
check("correction sharpens focus", opt["poly_rms_um"] < s_un["rms_um"])
check("diffractive reduces |LCA|", abs(opt["lca_D"]) < abs(lca_un["lca_D"]))

line("\n5. MTF smoke test")
f, m = geometric_mtf(eye0.system, 4.0, LAMS, nrays=8000, N=128)
line(f"   MTF at {f[3]:.0f} c/deg = {m[3]:.2f}; max freq {f[-1]:.0f} c/deg")
check("MTF starts near 1 and decreases", m[0] > 0.8 and m[-1] < m[0])

line("\nDone.")
