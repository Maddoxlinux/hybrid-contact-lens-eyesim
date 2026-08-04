# Ray-Tracing Model of Hybrid Diffractive–Refractive Contact Lenses

A Python ray-tracing model of the human eye (Navarro schematic eye) and a
**hybrid refractive–diffractive contact lens** for correcting **myopia /
hyperopia** while **reducing chromatic aberration**.

It reproduces the project proposal end-to-end: build & validate a normal eye,
introduce refractive errors, design the hybrid lens, run polychromatic ray
tracing, and compare the **normal / uncorrected / corrected** eye using
**spot diagrams, MTF and longitudinal chromatic aberration (LCA)**.

---

## What's inside

```
eyesim/            the simulation engine (pure NumPy)
  dispersion.py    wavelength-dependent ocular & lens indices (Cauchy)
  surfaces.py      conic refractive + diffractive surface primitives
  raytrace.py      3-D vector ray tracer + paraxial power engine
  eye.py           Navarro eye builder (parametric ametropia)
  lens.py          hybrid contact-lens builder + fitting helpers
  metrics.py       spot diagrams, geometric MTF, LCA
  optimize.py      numerical optimisation of the hybrid lens
  _opt.py          small dependency-free root-finder / simplex
app.py             interactive Streamlit web app
validate.py        physics sanity checks (run this first)
```

## Quick start (local)

```bash
pip install -r requirements.txt

python validate.py        # verify the physics
streamlit run app.py      # launch the interactive app -> http://localhost:8501
```

The engine itself needs only **numpy**; the web app adds **streamlit** and
**matplotlib**.

## Key validated results

| Quantity | Value |
|---|---|
| Emmetropic eye power / axial length | ~60.7 D / ~23.9 mm |
| Ocular LCA (450–650 nm), bare eye | ~1.7 D |
| −3 D myope, green blur (uncorrected → corrected) | ~83 µm → ~5 µm |
| Residual LCA after hybrid lens | ~0 D |

---

## Hosting

The app is a standard Streamlit app, so the simplest free option is
**Streamlit Community Cloud**:

1. Push this folder to a **public GitHub repo**.
2. Go to <https://share.streamlit.io>, sign in with GitHub.
3. **New app** → pick the repo, branch, and `app.py` → **Deploy**.
   It reads `requirements.txt` automatically and gives you a public URL.

**Alternative — Hugging Face Spaces:** create a new Space (SDK = *Streamlit*),
upload the folder (or connect the repo). Same `requirements.txt`.

> Vercel is optimised for JS/Next.js front-ends and is not the natural home for
> a long-running Streamlit process; use Streamlit Cloud or HF Spaces instead.

---

## Modelling notes & assumptions

- **Eye:** Navarro schematic eye (Escudero-Sanz & Navarro, *JOSA A* 1999),
  relaxed accommodation. Indices follow a two-term Cauchy law; Abbe numbers are
  tuned so the whole eye shows ~2 D of LCA (Thibos et al. 1992).
- **Ametropia** is modelled as axial (retina moved ~0.37 mm per dioptre).
- **Diffractive lens** is treated as a thin radial phase element blazed to
  order 1: effective power scales as `P_d(λ) = P_d0 · λ/λ0`, implemented as a
  momentum kick in the ray trace with a matching optical-path term.
- **MTF** is a geometric (ray-based) estimate, photopic-weighted for
  polychromatic light; a diffraction (Airy) reference is shown on the spots.
- With a contact lens on, the cornea faces the tear film, so the lens front
  surface takes over the air–cornea refraction — its "base power" therefore
  includes the corneal front power. The clinically meaningful outputs are the
  refractive error corrected and the diffractive add power.

These are geometric-optics approximations chosen for transparency and speed;
they capture the defocus, spherical and chromatic behaviour the project targets.
