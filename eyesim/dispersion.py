"""
Wavelength-dependent refractive indices of the ocular media and of the
contact-lens material.

Each medium is described by a two-term Cauchy law

    n(lambda) = A + B / lambda**2            (lambda in micrometres)

whose coefficients (A, B) are derived from the reference index at the
helium d-line (n_d, 587.6 nm) and the Abbe number V_d.  This reproduces
normal dispersion (n larger at short wavelengths) and, with the values
below, yields a total ocular longitudinal chromatic aberration of about
2 dioptres across the visible band -- the value measured in real eyes
(Thibos et al. 1992; Atchison & Smith, *Optics of the Human Eye*).

The n_d reference indices are those of the Navarro schematic eye
(Escudero-Sanz & Navarro, JOSA A 16, 1999).  Abbe numbers are chosen to
match the measured ocular chromatic difference of refraction; they are
kept here as an explicit, editable table so they can be checked against
whichever source a report cites.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# d, F and C spectral lines used to define the Abbe number (micrometres)
LAMBDA_D = 0.58756
LAMBDA_F = 0.48613
LAMBDA_C = 0.65627

# Convenient monochromatic design/reference wavelengths (micrometres)
LAMBDA_BLUE = 0.450
LAMBDA_GREEN = 0.550   # default design wavelength
LAMBDA_RED = 0.650


def cauchy_from_abbe(n_d: float, V_d: float) -> tuple[float, float]:
    """Return Cauchy (A, B) for a material given n_d and the Abbe number."""
    inv = 1.0 / LAMBDA_F**2 - 1.0 / LAMBDA_C**2
    B = (n_d - 1.0) / (V_d * inv)
    A = n_d - B / LAMBDA_D**2
    return A, B


def cauchy_index(lam_um, A: float, B: float):
    """Evaluate a Cauchy index at wavelength(s) `lam_um` (micrometres)."""
    lam = np.asarray(lam_um, dtype=float)
    return A + B / lam**2


@dataclass(frozen=True)
class Material:
    """A dispersive optical material."""
    name: str
    n_d: float
    V_d: float

    @property
    def _AB(self) -> tuple[float, float]:
        return cauchy_from_abbe(self.n_d, self.V_d)

    def index(self, lam_um):
        A, B = self._AB
        return cauchy_index(lam_um, A, B)


class OcularMedia:
    """Named collection of the four ocular media plus air.

    n_d values: Navarro schematic eye.  V_d values: tuned so that the whole
    eye shows ~2 D of longitudinal chromatic aberration over 450-650 nm.
    """
    AIR = Material("air", 1.0000, 1e9)          # effectively non-dispersive
    CORNEA = Material("cornea", 1.3760, 55.0)
    AQUEOUS = Material("aqueous", 1.3374, 52.0)
    LENS = Material("lens", 1.4201, 48.0)
    VITREOUS = Material("vitreous", 1.3360, 52.0)


# Common contact-lens materials (soft silicone-hydrogel and rigid PMMA).
LENS_MATERIALS = {
    "silicone-hydrogel": Material("silicone-hydrogel", 1.430, 45.0),
    "hydrogel": Material("hydrogel", 1.400, 50.0),
    "PMMA": Material("PMMA", 1.492, 57.4),
}


# ---------------------------------------------------------------------------
# Photopic luminous-efficiency function V(lambda), used to weight the
# polychromatic point-spread function.  Sampled every 10 nm, 400-700 nm
# (CIE 1924).  Interpolated on demand.
# ---------------------------------------------------------------------------
_PHOTOPIC_NM = np.arange(400, 701, 10)
_PHOTOPIC_V = np.array([
    0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910, 0.1390,
    0.2080, 0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520,
    0.8700, 0.7570, 0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070, 0.0610,
    0.0320, 0.0170, 0.0082, 0.0041,
])


def PHOTOPIC(lam_um):
    """Photopic weight V(lambda) for wavelength(s) given in micrometres."""
    lam_nm = np.asarray(lam_um, dtype=float) * 1000.0
    return np.interp(lam_nm, _PHOTOPIC_NM, _PHOTOPIC_V, left=0.0, right=0.0)
