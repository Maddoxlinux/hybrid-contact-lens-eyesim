"""
Ray-tracing engines.

`System` holds an ordered list of surfaces plus the material in front of the
first surface (object space, normally air).  Two engines act on a system:

* trace_rays  -- real 3-D sequential ray trace with the vector Snell law,
                 optical-path accumulation and diffractive momentum kicks.
* paraxial_power -- a first-order (y, nu) trace used to obtain the equivalent
                 power / back-focal distance of the system at a wavelength,
                 which drives the chromatic-aberration and focus calculations.

Object is at infinity (collimated input) unless stated otherwise, matching
the distance-vision case in the proposal.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from .dispersion import Material, OcularMedia
from .surfaces import Surface, ConicSurface, DiffractiveSurface, ImagePlane


@dataclass
class System:
    surfaces: list[Surface]
    material_before: Material = OcularMedia.AIR

    def index_before(self, i: int, lam_um) -> float:
        """Refractive index of the medium immediately before surface i."""
        if i == 0:
            return float(self.material_before.index(lam_um))
        return float(self.surfaces[i - 1].material_after.index(lam_um))

    @property
    def image_plane_z(self) -> float:
        return self.surfaces[-1].z


def _refract(d, n, mu):
    """Vector Snell's law.  d, n unit vectors; mu = n1/n2.  Returns unit dir."""
    # orient normal against the incident ray
    cosi = -np.sum(d * n, axis=-1, keepdims=True)
    n = np.where(cosi < 0, -n, n)
    cosi = np.abs(cosi)
    sint2 = mu * mu * (1.0 - cosi * cosi)
    tir = sint2 > 1.0
    cost = np.sqrt(np.maximum(1.0 - sint2, 0.0))
    dt = mu * d + (mu * cosi - cost) * n
    dt = dt / np.linalg.norm(dt, axis=-1, keepdims=True)
    return dt, tir[..., 0]


def trace_rays(system: System, x0, y0, lam_um, image_z: float | None = None):
    """Trace a bundle of collimated axial rays through `system`.

    Parameters
    ----------
    x0, y0 : arrays of entrance-pupil coordinates (mm) at the first surface.
    lam_um : wavelength (micrometres).
    image_z : optional plane to record the final intersection at; defaults to
              the system's last surface (the retina).

    Returns a dict with per-ray final coordinates on the image plane, the
    accumulated optical path length, and a boolean validity mask.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    y0 = np.asarray(y0, dtype=float).ravel()
    nrays = x0.size

    # start on a common plane well in front of the first surface so that all
    # rays share the same initial phase (planar wavefront, object at infinity)
    z_start = system.surfaces[0].z - 5.0
    p = np.stack([x0, y0, np.full(nrays, z_start)], axis=-1)
    d = np.tile(np.array([0.0, 0.0, 1.0]), (nrays, 1))
    opl = np.zeros(nrays)
    valid = np.ones(nrays, dtype=bool)

    n_before = system.index_before(0, lam_um)

    for i, surf in enumerate(system.surfaces):
        t = surf.intersect(p, d)
        ok = np.isfinite(t) & (t > 0)
        valid &= ok
        t = np.where(ok, t, 0.0)
        p = p + t[:, None] * d
        opl += n_before * t

        # aperture check
        r = np.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2)
        valid &= r <= surf.semi_diameter + 1e-9

        n_after = float(surf.material_after.index(lam_um))

        if isinstance(surf, ImagePlane):
            break

        if isinstance(surf, ConicSurface):
            # refract at the interface (flat surfaces have R = inf)
            nrm = surf.normal(p)
            mu = n_before / n_after
            d, tir = _refract(d, nrm, mu)
            valid &= ~tir

        if isinstance(surf, DiffractiveSurface):
            d, add_opl = surf.diffract(p, d, n_after, lam_um)
            opl += add_opl

        n_before = n_after

    # propagate to requested image plane if different from last surface
    if image_z is not None and abs(image_z - p[..., 2]).max() > 1e-9:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (image_z - p[..., 2]) / d[..., 2]
        p = p + t[:, None] * d
        opl += n_before * t

    return {
        "x": p[..., 0], "y": p[..., 1], "z": p[..., 2],
        "opl": opl, "valid": valid, "dir": d,
    }


def paraxial_power(system: System, lam_um):
    """First-order equivalent power (dioptres) and back-focal distance (mm).

    Traces a marginal ray from an axial object at infinity (y = 1, angle 0)
    using the reduced-angle (y, omega = n*u) convention.  Diffractive
    surfaces contribute their wavelength-scaled add power.
    """
    y = 1.0
    omega = 0.0                       # n*u ; collimated -> u = 0
    n_before = system.index_before(0, lam_um)

    zs = [s.z for s in system.surfaces]
    for i, surf in enumerate(system.surfaces):
        if isinstance(surf, ImagePlane):
            break
        n_after = float(surf.material_after.index(lam_um))

        # transfer from previous surface to this one
        if i > 0:
            t = zs[i] - zs[i - 1]
            u = omega / n_before
            y = y + u * t

        # refractive power of the surface
        if np.isfinite(getattr(surf, "R", np.inf)):
            K = (n_after - n_before) / surf.R          # 1/mm
        else:
            K = 0.0
        omega = omega - y * K

        # diffractive add power (1/mm), scaled with wavelength
        if isinstance(surf, DiffractiveSurface):
            Kd = surf.add_power(lam_um) / 1000.0
            omega = omega - y * Kd

        n_before = n_after

    # equivalent power referred to the first surface: Phi = -omega_final / y0
    Phi_permm = -omega / 1.0
    power_D = Phi_permm * 1000.0
    bfd = np.inf if Phi_permm == 0 else (y / (-omega)) * n_before  # mm to focus
    return power_D, bfd
