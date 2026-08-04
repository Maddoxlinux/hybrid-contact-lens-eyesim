# Vercel app — Hybrid Contact-Lens Eye Simulator

A static single-page frontend + a **Python serverless function** that runs the
`eyesim` ray-tracing engine and returns spot / MTF / chromatic-aberration data
as JSON. No build step.

```
vercel-app/
  index.html          single-page UI (vanilla JS + SVG charts)
  api/
    simulate.py       Vercel Python function  (POST /api/simulate)
    _compute.py       framework-free compute layer (unit-testable)
    eyesim/           vendored copy of the engine (imported by the function)
  requirements.txt    numpy
  vercel.json         function memory / duration
```

## Preview locally (no Vercel needed)

```bash
cd vercel-app
pip install numpy
python devserver.py       # serves page + API together
```

Open <http://localhost:8000>. (The 3-D view loads Three.js from a CDN, so the
browser needs internet; all the physics runs locally.)

## Deploy on Vercel

Everything is already in the GitHub repo, so:

1. Go to <https://vercel.com/new> and **import** the repo
   `Maddoxlinux/hybrid-contact-lens-eyesim`.
2. Set **Root Directory** to `vercel-app`.
3. Framework preset: **Other** (it's static + a Python function — no build needed).
4. **Deploy.** Vercel installs `numpy` from `requirements.txt`, serves
   `index.html`, and exposes the function at `/api/simulate`.

You get a URL like `https://<project>.vercel.app`.

> The `eyesim` engine is vendored under `api/` so the serverless function
> bundles it. The source of truth is the top-level `eyesim/` package (used by
> the Streamlit app); re-copy it here if you change the engine.

## API

`POST /api/simulate` with JSON:

```json
{ "error_D": -3.0, "pupil": 4.0, "use_lens": true, "use_diffractive": true,
  "material": "silicone-hydrogel", "design_lam_nm": 550, "mode": "Auto-fit" }
```

Returns `{ params, lens, states: { normal, uncorrected, corrected } }`, each
state holding `spots` (per wavelength), `mtf`, `lca`, `rms_um`, `lca_D`.
`GET /api/simulate` runs with defaults (health check).
