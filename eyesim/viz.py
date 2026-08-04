"""
Geometry and ray-path export for 3-D / cross-section visualisation.

`system_geometry` returns the optical surfaces (position, radius, conic,
aperture) so a frontend can draw the eye and the contact lens.  `trace_paths`
re-runs the real sequential ray trace but records every vertex, so the exact
ray polylines (including the diffractive bend) can be drawn.  Rays are
meridional (x = 0, in the y-z plane); by rotational symmetry they stay in that
plane, which is all that is needed to show focusing and chromatic spread.
"""
from __future__ import annotations
import numpy as np

from .surfaces import ConicSurface, DiffractiveSurface, ImagePlane
from .raytrace import System, _refract


def system_geometry(system: System) -> list[dict]:
    out = []
    for s in system.surfaces:
        R = getattr(s, "R", np.inf)
        out.append(dict(
            z=round(float(s.z), 4),
            R=(None if not np.isfinite(R) else round(float(R), 4)),
            k=round(float(getattr(s, "k", 0.0)), 4),
            sd=round(float(s.semi_diameter), 4),
            kind=type(s).__name__,
            name=s.name,
        ))
    return out


def _sag_profile(R, k, sd, n=24):
    """Return (z_offset, y) points of a conic surface profile, y in [-sd, sd]."""
    ys = np.linspace(-sd, sd, n)
    if R is None or not np.isfinite(R):
        z = np.zeros_like(ys)
    else:
        c = 1.0 / R
        z = (c * ys ** 2) / (1.0 + np.sqrt(np.maximum(1.0 - (1.0 + k) * c * c * ys ** 2, 0.0)))
    return [[round(float(zz), 4), round(float(yy), 4)] for zz, yy in zip(z, ys)]


def surface_profiles(system: System, n=24) -> list[dict]:
    """Sag profiles for drawing each surface as a curved line / lathe."""
    prof = []
    for s in system.surfaces:
        R = getattr(s, "R", np.inf)
        prof.append(dict(
            z=round(float(s.z), 4), sd=round(float(s.semi_diameter), 4),
            kind=type(s).__name__, name=s.name,
            profile=_sag_profile(None if not np.isfinite(R) else R,
                                 getattr(s, "k", 0.0), s.semi_diameter, n),
        ))
    return prof


def trace_paths(system: System, ys, lam_um) -> list[list]:
    """Trace meridional rays at heights `ys`; return each ray as [[z, y], ...]."""
    ys = np.atleast_1d(np.asarray(ys, dtype=float))
    n_first = system.index_before(0, lam_um)
    z_start = system.surfaces[0].z - 3.0
    rays = []
    for y0 in ys:
        p = np.array([[0.0, float(y0), z_start]])
        d = np.array([[0.0, 0.0, 1.0]])
        pts = [[round(z_start, 4), round(float(y0), 4)]]
        n_before = n_first
        for i, surf in enumerate(system.surfaces):
            t = surf.intersect(p, d)[0]
            if not np.isfinite(t) or t <= 1e-9:
                break
            p = p + t * d
            pts.append([round(float(p[0, 2]), 4), round(float(p[0, 1]), 4)])
            n_after = float(surf.material_after.index(lam_um))
            if isinstance(surf, ImagePlane):
                break
            if isinstance(surf, ConicSurface):
                nrm = surf.normal(p)
                d, tir = _refract(d, nrm, n_before / n_after)
                if tir[0]:
                    break
            if isinstance(surf, DiffractiveSurface):
                d, _ = surf.diffract(p, d, n_after, lam_um)
            n_before = n_after
        rays.append(pts)
    return rays
