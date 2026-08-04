"""
Optical-performance metrics: spot diagrams, geometric MTF and longitudinal
chromatic aberration.

All three operate on a `System` whose last surface is the retina.

* spot_diagram -- ray intersections on the retina, referenced to their
                  centroid, plus the RMS and geometric spot radius.
* geometric_mtf -- modulation transfer function obtained from the ray-based
                  point-spread function (a 2-D histogram of ray hits).  For
                  polychromatic light the per-wavelength PSFs are summed with
                  photopic weights.  Frequencies are returned in cycles/degree
                  using a posterior nodal distance of 16.7 mm (0.2914 mm/deg).
* longitudinal_chromatic_aberration -- equivalent power and focus position as
                  a function of wavelength (the classic ~2 D ocular LCA curve).

These are geometric-optics estimates; at the pupil sizes of interest they
capture the defocus, spherical and chromatic behaviour that the project set
out to quantify.
"""

from __future__ import annotations
import numpy as np

from .dispersion import PHOTOPIC
from .raytrace import System, trace_rays, paraxial_power

MM_PER_DEG = 16.7 * np.tan(np.deg2rad(1.0))    # retinal scale, ~0.2915 mm/deg


# --------------------------------------------------------------------------
# pupil sampling
# --------------------------------------------------------------------------
def hexapolar_pupil(semi: float, rings: int = 12):
    """A hexapolar grid of points filling a disk of radius `semi`."""
    pts = [(0.0, 0.0)]
    for i in range(1, rings + 1):
        r = semi * i / rings
        n = 6 * i
        for j in range(n):
            a = 2.0 * np.pi * j / n
            pts.append((r * np.cos(a), r * np.sin(a)))
    p = np.array(pts)
    return p[:, 0], p[:, 1]


def random_pupil(semi: float, n: int, seed: int = 0):
    """`n` uniform random samples over a disk of radius `semi`."""
    rng = np.random.default_rng(seed)
    r = semi * np.sqrt(rng.random(n))
    a = 2.0 * np.pi * rng.random(n)
    return r * np.cos(a), r * np.sin(a)


# --------------------------------------------------------------------------
# spot diagram
# --------------------------------------------------------------------------
def spot_diagram(system: System, pupil_diameter: float, lam_um,
                 rings: int = 14):
    """Return dict with ray x,y on the retina (microns) and spot metrics."""
    semi = pupil_diameter / 2.0
    x0, y0 = hexapolar_pupil(semi, rings)
    res = trace_rays(system, x0, y0, lam_um, image_z=system.image_plane_z)
    v = res["valid"]
    x, y = res["x"][v], res["y"][v]
    if x.size == 0:
        return dict(x=x, y=y, rms_um=np.nan, geo_um=np.nan, cx=np.nan, cy=np.nan)
    cx, cy = x.mean(), y.mean()
    dx, dy = x - cx, y - cy
    rms = np.sqrt(np.mean(dx * dx + dy * dy)) * 1000.0     # microns
    geo = np.sqrt(np.max(dx * dx + dy * dy)) * 1000.0      # microns (max)
    return dict(x=dx * 1000.0, y=dy * 1000.0, rms_um=rms, geo_um=geo,
                cx=cx, cy=cy)


# --------------------------------------------------------------------------
# geometric MTF
# --------------------------------------------------------------------------
def _psf_histogram(system, semi, lam_um, fov_mm, N, nrays, seed):
    x0, y0 = random_pupil(semi, nrays, seed=seed)
    res = trace_rays(system, x0, y0, lam_um, image_z=system.image_plane_z)
    v = res["valid"]
    x, y = res["x"][v], res["y"][v]
    if x.size == 0:
        return np.zeros((N, N))
    cx, cy = x.mean(), y.mean()
    edges = np.linspace(-fov_mm / 2, fov_mm / 2, N + 1)
    H, _, _ = np.histogram2d(x - cx, y - cy, bins=[edges, edges])
    return H


def geometric_mtf(system: System, pupil_diameter: float, wavelengths,
                  weights=None, fov_mm: float | None = None, N: int = 256,
                  nrays: int = 40000, seed: int = 1):
    """Polychromatic geometric MTF.

    Returns (freq_cyc_per_deg, mtf) sampled from 0 up to the Nyquist limit
    of the PSF grid.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if weights is None:
        weights = PHOTOPIC(wavelengths)
        if weights.sum() == 0:
            weights = np.ones_like(wavelengths)
    weights = weights / weights.sum()
    semi = pupil_diameter / 2.0

    if fov_mm is None:
        # size the field from the worst-case spot so the PSF is not clipped
        worst = 0.0
        for lam in wavelengths:
            s = spot_diagram(system, pupil_diameter, lam, rings=8)
            worst = max(worst, (s["geo_um"] or 0.0) / 1000.0)
        fov_mm = max(0.02, 2.5 * worst)

    psf = np.zeros((N, N))
    for lam, w in zip(wavelengths, weights):
        psf += w * _psf_histogram(system, semi, lam, fov_mm, N, nrays, seed)
    if psf.sum() == 0:
        f = np.linspace(0, 60, 100)
        return f, np.zeros_like(f)
    psf /= psf.sum()

    otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
    mtf2d = np.abs(otf)
    mtf2d /= mtf2d[N // 2, N // 2]

    # radial average
    pixel_mm = fov_mm / N
    f_cyc_mm = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_mm))
    fx, fy = np.meshgrid(f_cyc_mm, f_cyc_mm)
    fr = np.sqrt(fx * fx + fy * fy)
    fmax = f_cyc_mm.max()
    nb = N // 2
    bins = np.linspace(0, fmax, nb + 1)
    idx = np.digitize(fr.ravel(), bins) - 1
    mtf = np.zeros(nb)
    m = mtf2d.ravel()
    for k in range(nb):
        sel = idx == k
        mtf[k] = m[sel].mean() if sel.any() else np.nan
    centers = 0.5 * (bins[:-1] + bins[1:])
    freq_cyc_deg = centers * MM_PER_DEG
    good = np.isfinite(mtf)
    return freq_cyc_deg[good], np.clip(mtf[good], 0.0, 1.0)


# --------------------------------------------------------------------------
# longitudinal chromatic aberration
# --------------------------------------------------------------------------
def longitudinal_chromatic_aberration(system: System, wavelengths,
                                      lam_ref_um: float = 0.550):
    """Return dict of arrays: power(D), focus_shift(mm) and LCA vs wavelength.

    Focus and power are referenced to the value at `lam_ref_um`.  The scalar
    `lca_D` is power(shortest) - power(longest).
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    # exclude the retina (last surface) for the power calculation
    sys_refr = System(system.surfaces[:-1], system.material_before)
    powers = np.array([paraxial_power(sys_refr, lam)[0] for lam in wavelengths])
    bfds = np.array([paraxial_power(sys_refr, lam)[1] for lam in wavelengths])
    P_ref = np.interp(lam_ref_um, wavelengths, powers)
    b_ref = np.interp(lam_ref_um, wavelengths, bfds)
    lca_D = float(powers[np.argmin(wavelengths)] - powers[np.argmax(wavelengths)])
    return dict(wavelengths=wavelengths, power_D=powers,
                power_rel_D=powers - P_ref,
                focus_mm=bfds, focus_shift_mm=bfds - b_ref,
                lca_D=lca_D)
