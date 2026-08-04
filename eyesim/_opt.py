"""
Tiny dependency-free optimisers (so the engine needs only numpy).

* bisect       -- bracketed root finder for a scalar function.
* nelder_mead  -- downhill-simplex minimiser for a few parameters.
"""

from __future__ import annotations
import numpy as np


def bisect(f, lo, hi, xtol=1e-4, maxiter=100):
    """Root of f on [lo, hi] assuming a sign change; else nearest-to-zero."""
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        xs = np.linspace(lo, hi, 201)
        vals = np.array([f(x) for x in xs])
        return float(xs[int(np.argmin(np.abs(vals)))])
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < 1e-12 or (hi - lo) < xtol:
            return float(mid)
        if np.sign(fmid) == np.sign(flo):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return float(0.5 * (lo + hi))


def nelder_mead(f, x0, step=0.5, xatol=1e-3, fatol=1e-2, maxiter=400):
    """Minimal Nelder-Mead simplex minimiser.  x0 is a sequence of floats."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    # build initial simplex
    sim = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] += step if p[i] == 0 else step * (1 + abs(p[i]))
        sim.append(p)
    sim = np.array(sim)
    fv = np.array([f(p) for p in sim])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(maxiter):
        order = np.argsort(fv)
        sim, fv = sim[order], fv[order]
        if (np.max(np.abs(sim[1:] - sim[0])) <= xatol and
                np.max(np.abs(fv[1:] - fv[0])) <= fatol):
            break
        centroid = sim[:-1].mean(axis=0)
        xr = centroid + alpha * (centroid - sim[-1])
        fr = f(xr)
        if fv[0] <= fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        elif fr < fv[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                sim[-1], fv[-1] = xe, fe
            else:
                sim[-1], fv[-1] = xr, fr
        else:
            xc = centroid + rho * (sim[-1] - centroid)
            fc = f(xc)
            if fc < fv[-1]:
                sim[-1], fv[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    sim[i] = sim[0] + sigma * (sim[i] - sim[0])
                    fv[i] = f(sim[i])
    order = np.argsort(fv)
    return sim[order][0], fv[order][0]
