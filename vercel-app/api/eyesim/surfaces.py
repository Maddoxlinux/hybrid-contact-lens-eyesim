"""
Optical surface primitives.

A `System` (see raytrace.py) is an ordered list of surfaces, each carrying
the material that follows it.  Three surface types are provided:

* ConicSurface      -- rotationally symmetric conic (sphere when k = 0),
                       refraction by the vector Snell law.
* DiffractiveSurface -- a ConicSurface that additionally carries a radial
                       diffractive phase profile (a kinoform / diffractive
                       lens).  Its added optical power scales linearly with
                       wavelength, which is what lets it correct the chromatic
                       aberration of the refractive parts of the system.
* ImagePlane        -- a flat detector (the retina).

Sign conventions (standard, light travels toward +z):
    radius R  > 0  -> centre of curvature to the right of the vertex
    conic  k       -> k = -e**2 ; sphere k = 0, paraboloid k = -1
    powers positive = converging
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .dispersion import Material


@dataclass
class Surface:
    """Base surface. `z` is the vertex position on the optical axis (mm)."""
    z: float
    material_after: Material          # medium immediately downstream
    semi_diameter: float = 6.0        # clear aperture radius (mm)
    name: str = ""

    # ---- geometry, overridden by subclasses -----------------------------
    def intersect(self, p, d):
        """Return distance t along unit dir d from point p to the surface."""
        raise NotImplementedError

    def normal(self, p):
        """Outward surface normal (unit) at surface point p."""
        raise NotImplementedError


@dataclass
class ConicSurface(Surface):
    R: float = np.inf                 # radius of curvature (mm), inf = plane
    k: float = 0.0                    # conic constant

    def sag(self, r2):
        """Sag z(r) measured from the vertex, for r**2 = x**2 + y**2."""
        if not np.isfinite(self.R):
            return np.zeros_like(r2)
        c = 1.0 / self.R
        return (c * r2) / (1.0 + np.sqrt(np.maximum(1.0 - (1.0 + self.k) * c * c * r2, 0.0)))

    def intersect(self, p, d):
        # Conic surface (vertex at z=self.z) as implicit surface
        #   x^2 + y^2 + (1+k)(Z)^2 - 2 R Z = 0 ,  Z = z - z_vertex
        x0, y0, z0 = p[..., 0], p[..., 1], p[..., 2] - self.z
        L, M, N = d[..., 0], d[..., 1], d[..., 2]

        if not np.isfinite(self.R):
            # plane z = z_vertex
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(np.abs(N) > 1e-12, (self.z - p[..., 2]) / N, np.nan)
            return t

        K = 1.0 + self.k
        R = self.R
        a = L * L + M * M + K * N * N
        b = 2.0 * (x0 * L + y0 * M + K * z0 * N - R * N)
        c = x0 * x0 + y0 * y0 + K * z0 * z0 - 2.0 * R * z0

        disc = b * b - 4.0 * a * c
        valid = disc >= 0
        sq = np.sqrt(np.where(valid, disc, 0.0))
        # two roots; choose the physically forward one nearest the vertex
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-b - sq) / (2.0 * a)
            t2 = (-b + sq) / (2.0 * a)
        # pick the smaller positive root
        t = np.where((t1 > 1e-9), t1, t2)
        t = np.where(valid & (t > 1e-9), t, np.nan)
        return t

    def normal(self, p):
        if not np.isfinite(self.R):
            n = np.zeros_like(p)
            n[..., 2] = 1.0
            return n
        Z = p[..., 2] - self.z
        grad = np.stack([p[..., 0], p[..., 1], (1.0 + self.k) * Z - self.R], axis=-1)
        norm = np.linalg.norm(grad, axis=-1, keepdims=True)
        return grad / np.where(norm == 0, 1.0, norm)


@dataclass
class DiffractiveSurface(ConicSurface):
    """Conic surface carrying a radial diffractive phase profile.

    The design phase (at the design wavelength `lam0_um`) is

        Phi(r) = -(2*pi / lam0) * ( 0.5 * P_d0/1000 * r**2 + a4 * r**4 )

    where `P_d0` is the diffractive add power in dioptres at the design
    wavelength and `a4` an optional 4th-order coefficient for spherical
    aberration control.  Tracing a real ray in diffraction order `m`
    (blazed to m = 1) adds a transverse slope proportional to the local
    phase gradient, giving an effective power  P_d(lam) = P_d0 * lam/lam0.
    The equivalent optical-path term  m*lam/(2*pi) * Phi(r)  is added to
    the accumulated path length so that the wavefront (and hence the MTF)
    stays self-consistent with the ray bending.
    """
    P_d0: float = 0.0                 # diffractive add power at lam0 (dioptres)
    a4: float = 0.0                   # 4th-order phase coefficient (1/mm^3)
    lam0_um: float = 0.550            # design wavelength (micrometres)
    order: int = 1                    # diffraction order the lens is blazed to

    def _phase_and_grad(self, r2, lam_um):
        """Return (Phi, dPhi/d(r) / r) at radius^2 = r2 for wavelength lam."""
        lam0_mm = self.lam0_um * 1e-3
        # P_d0 is in dioptres (1/m) -> 1/mm is /1000
        c2 = -(2.0 * np.pi / lam0_mm) * 0.5 * (self.P_d0 / 1000.0)
        c4 = -(2.0 * np.pi / lam0_mm) * self.a4
        Phi = c2 * r2 + c4 * r2 * r2
        # d Phi / dr = 2 c2 r + 4 c4 r^3  ->  (d Phi/dr)/r = 2 c2 + 4 c4 r^2
        dPhi_over_r = 2.0 * c2 + 4.0 * c4 * r2
        return Phi, dPhi_over_r

    def diffract(self, p, d, n_after, lam_um):
        """Apply the diffractive momentum kick to already-refracted dir `d`.

        Returns (new_direction, added_optical_path_mm).
        """
        lam_mm = lam_um * 1e-3
        r2 = p[..., 0] ** 2 + p[..., 1] ** 2
        Phi, dPhi_over_r = self._phase_and_grad(r2, lam_um)
        # transverse gradient vector = (dPhi/dr)/r * (x, y)
        gx = dPhi_over_r * p[..., 0]
        gy = dPhi_over_r * p[..., 1]
        # added transverse direction-cosine change: (m*lam/2pi) * grad / n_after
        fac = self.order * lam_mm / (2.0 * np.pi) / n_after
        L = d[..., 0] + fac * gx
        M = d[..., 1] + fac * gy
        N2 = 1.0 - L * L - M * M
        Nsign = np.sign(d[..., 2])
        N = Nsign * np.sqrt(np.maximum(N2, 0.0))
        newd = np.stack([L, M, N], axis=-1)
        newd /= np.linalg.norm(newd, axis=-1, keepdims=True)
        added_opl = (self.order * lam_mm / (2.0 * np.pi)) * Phi
        return newd, added_opl

    def add_power(self, lam_um):
        """Effective diffractive power (dioptres) at wavelength lam."""
        return self.P_d0 * (lam_um / self.lam0_um)


@dataclass
class ImagePlane(ConicSurface):
    """A flat detector plane (the retina)."""
    def __post_init__(self):
        self.R = np.inf
        self.k = 0.0
