"""
Hybrid refractive-diffractive contact lens.

The lens has two refracting surfaces (a diffractive front surface and a
spherical back surface that fits the cornea) separated by a thin body of
lens material, resting on a thin tear film in front of the cornea.

* The **refractive** base provides the bulk optical power that pushes the
  ametropic eye's focus back onto the retina.
* The **diffractive** phase profile on the front surface adds a power that
  grows with wavelength (P_d(lambda) = P_d0 * lambda/lambda0).  Because the
  refractive surfaces of the eye lose power with wavelength, a suitably
  chosen P_d0 makes the *total* system power nearly wavelength-independent,
  i.e. it cancels the longitudinal chromatic aberration.

Two fitting helpers are provided:
    fit_base_power        -> base power that corrects the spherical error
    fit_diffractive_power -> P_d0 that minimises the ocular chromatic aberration
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ._opt import bisect
from .dispersion import OcularMedia, LENS_MATERIALS, Material, LAMBDA_GREEN, \
    LAMBDA_BLUE, LAMBDA_RED
from .surfaces import ConicSurface, DiffractiveSurface, ImagePlane
from .raytrace import System, paraxial_power

TEAR = OcularMedia.AQUEOUS          # tear film modelled with the aqueous index


@dataclass
class ContactLens:
    refractive_power_D: float
    diffractive_power_D: float
    design_lam_um: float
    material: Material
    R_back: float
    R_front: float
    thickness: float
    a4: float
    surfaces: list


def _front_radius(power_D, material, R_back, gap_to_tear=True):
    """Solve the thin-lens equation for the front radius (mm)."""
    n = material.n_d
    n_tear = TEAR.n_d
    # P/1000 = (n-1)/R_front + (n_tear - n)/R_back   [1/mm]
    rhs = power_D / 1000.0 - (n_tear - n) / R_back
    if abs(rhs) < 1e-12:
        return np.inf
    return (n - 1.0) / rhs


def build_hybrid_lens(refractive_power_D: float,
                      diffractive_power_D: float = 0.0,
                      design_lam_um: float = LAMBDA_GREEN,
                      material: str | Material = "silicone-hydrogel",
                      R_back: float = 7.80,
                      thickness: float = 0.15,
                      tear_gap: float = 0.05,
                      semi_diameter: float = 6.0,
                      a4: float = 0.0) -> ContactLens:
    """Build a hybrid contact lens sitting in front of the cornea (z < 0)."""
    mat = LENS_MATERIALS[material] if isinstance(material, str) else material
    R_front = _front_radius(refractive_power_D, mat, R_back)

    z_back = -tear_gap
    z_front = z_back - thickness

    front = DiffractiveSurface(
        z=z_front, material_after=mat, semi_diameter=semi_diameter,
        R=R_front, k=0.0, P_d0=diffractive_power_D, a4=a4,
        lam0_um=design_lam_um, name="cl-front-diffractive")
    back = ConicSurface(
        z=z_back, material_after=TEAR, semi_diameter=semi_diameter,
        R=R_back, k=0.0, name="cl-back")

    return ContactLens(refractive_power_D=refractive_power_D,
                       diffractive_power_D=diffractive_power_D,
                       design_lam_um=design_lam_um, material=mat,
                       R_back=R_back, R_front=R_front, thickness=thickness,
                       a4=a4, surfaces=[front, back])


# --------------------------------------------------------------------------
# Fitting helpers
# --------------------------------------------------------------------------
def _system_with_lens(eye_builder, refr_power, diff_power, design_lam,
                      material, pupil, lam_ref):
    lens = build_hybrid_lens(refr_power, diff_power, design_lam, material)
    eye = eye_builder(front_optics=lens.surfaces)
    return eye, lens


def fit_base_power(eye_builder, pupil, lam_ref=LAMBDA_GREEN,
                   material="silicone-hydrogel", design_lam=LAMBDA_GREEN):
    """Return the refractive base power (D) that focuses on the retina.

    `eye_builder(front_optics=...)` must return an EyeModel whose retina_z is
    fixed by the eye's anatomy.  We root-find the base power for which the
    paraxial image coincides with the retina at the reference wavelength.
    """
    ref_eye = eye_builder(front_optics=None)
    retina_z = ref_eye.retina_z
    z_post_lens = ref_eye.system.surfaces[-2].z   # posterior lens surface

    def defocus(P):
        lens = build_hybrid_lens(P, 0.0, design_lam, material)
        eye = eye_builder(front_optics=lens.surfaces)
        _, bfd = paraxial_power(
            System(eye.system.surfaces[:-1], eye.system.material_before), lam_ref)
        image_z = z_post_lens + bfd
        return image_z - retina_z

    # bracket: correcting powers rarely exceed +/- 20 D
    return bisect(defocus, -25.0, 25.0, xtol=1e-4)


def fit_diffractive_power(eye_builder, base_power, pupil,
                          material="silicone-hydrogel", design_lam=LAMBDA_GREEN,
                          lam_a=LAMBDA_BLUE, lam_b=LAMBDA_RED):
    """Return P_d0 (D) that equalises system power at two wavelengths.

    Longitudinal chromatic aberration is the difference in equivalent power
    between short and long wavelengths.  With the diffractive term this
    difference is linear in P_d0, so two evaluations locate the zero.
    """
    def lca(Pd):
        lens = build_hybrid_lens(base_power, Pd, design_lam, material)
        eye = eye_builder(front_optics=lens.surfaces)
        sys_refr = System(eye.system.surfaces[:-1], eye.system.material_before)
        Pa, _ = paraxial_power(sys_refr, lam_a)
        Pb, _ = paraxial_power(sys_refr, lam_b)
        return Pa - Pb            # power(blue) - power(red)

    f0 = lca(0.0)
    f1 = lca(1.0)
    slope = f1 - f0
    if abs(slope) < 1e-9:
        return 0.0
    return float(-f0 / slope)
