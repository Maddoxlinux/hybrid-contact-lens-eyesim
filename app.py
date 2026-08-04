"""
Interactive web app for the hybrid refractive-diffractive contact-lens study.

Run locally:   streamlit run app.py
"""
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from eyesim.eye import build_navarro_eye
from eyesim.lens import build_hybrid_lens, fit_base_power, fit_diffractive_power
from eyesim.metrics import (spot_diagram, geometric_mtf,
                            longitudinal_chromatic_aberration, MM_PER_DEG)
from eyesim.optimize import optimize_hybrid_lens
from eyesim.dispersion import PHOTOPIC, LAMBDA_GREEN

st.set_page_config(page_title="Hybrid Contact-Lens Eye Simulator",
                   page_icon="👁️", layout="wide")

# wavelength palette for spot plots
SPOT_LAMS = [(0.450, "#3b5bff"), (0.550, "#2ca02c"), (0.650, "#e23b3b")]
LCA_LAMS = np.linspace(0.45, 0.65, 9)


# --------------------------------------------------------------------------
# cached computations
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_states(error_D, pupil, material, design_lam, use_lens,
               use_diffractive, mode, man_base, man_diff, opt_a4):
    """Return the three eye states and the fitted lens parameters."""
    normal = build_navarro_eye(0.0, pupil, LAMBDA_GREEN)
    uncorr = build_navarro_eye(error_D, pupil, LAMBDA_GREEN)

    info = {}
    if not use_lens:
        return normal, uncorr, uncorr, info

    builder = partial(build_navarro_eye, error_D, pupil_diameter=pupil,
                      lam_ref_um=LAMBDA_GREEN)

    if mode == "Manual":
        base = man_base
        diff = man_diff if use_diffractive else 0.0
        a4 = 0.0
    elif mode == "Auto-fit":
        base = fit_base_power(builder, pupil, design_lam, material, design_lam)
        diff = (fit_diffractive_power(builder, base, pupil, material, design_lam)
                if use_diffractive else 0.0)
        a4 = 0.0
    else:  # Optimize
        res = optimize_hybrid_lens(builder, pupil, design_lam, material,
                                   wavelengths=LCA_LAMS[::2], refine_a4=opt_a4)
        base = res["refractive_power_D"]
        diff = res["diffractive_power_D"] if use_diffractive else 0.0
        a4 = res["a4"]

    lens = build_hybrid_lens(base, diff, design_lam, material, a4=a4)
    corrected = build_navarro_eye(error_D, pupil, LAMBDA_GREEN,
                                  front_optics=lens.surfaces)
    info = dict(base=base, diff=diff, a4=a4, R_front=lens.R_front,
                R_back=lens.R_back)
    return normal, uncorr, corrected, info


@st.cache_data(show_spinner=False)
def mtf_curve(_system, pupil, tag):
    f, m = geometric_mtf(_system, pupil, LCA_LAMS[::2], nrays=16000, N=192)
    return f, m


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def plot_spots(states, pupil, span_um):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    titles = ["Normal eye", "Uncorrected", "Corrected"]
    for ax, sysobj, title in zip(axes, states, titles):
        for lam, color in SPOT_LAMS:
            s = spot_diagram(sysobj.system, pupil, lam, rings=12)
            ax.scatter(s["x"], s["y"], s=3, color=color, alpha=0.6,
                       label=f"{lam*1000:.0f} nm")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-span_um, span_um); ax.set_ylim(-span_um, span_um)
        ax.set_aspect("equal"); ax.set_xlabel("µm"); ax.grid(alpha=0.25)
        # Airy reference
        airy = 1.22 * 0.55e-3 * 16.7 / pupil * 1000
        ax.add_patch(plt.Circle((0, 0), airy, fill=False, ls="--",
                                 color="0.4", lw=0.8))
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.6)
    fig.tight_layout()
    return fig


def plot_mtf(states, pupil):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels = ["Normal eye", "Uncorrected", "Corrected"]
    colors = ["#555555", "#e23b3b", "#2ca02c"]
    for sysobj, lab, c in zip(states, labels, colors):
        f, m = mtf_curve(sysobj.system, pupil, lab)
        ax.plot(f, m, label=lab, color=c, lw=2)
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Spatial frequency (cycles/degree)")
    ax.set_ylabel("Modulation (MTF)")
    ax.set_title("Polychromatic MTF (photopic-weighted)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def plot_lca(uncorr, corrected, use_lens):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for sysobj, lab, c in [(uncorr, "Uncorrected eye", "#e23b3b"),
                           (corrected, "Corrected (hybrid lens)", "#2ca02c")]:
        d = longitudinal_chromatic_aberration(sysobj.system, LCA_LAMS,
                                               LAMBDA_GREEN)
        ax.plot(d["wavelengths"] * 1000, d["power_rel_D"], "o-", label=lab,
                color=c, lw=2)
        if not use_lens and lab.startswith("Corrected"):
            break
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative equivalent power (D)\n(referenced to 550 nm)")
    ax.set_title("Longitudinal chromatic aberration")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# sidebar controls
# --------------------------------------------------------------------------
st.sidebar.title("👁️ Eye & Lens controls")

error_D = st.sidebar.slider("Refractive error (D)  —  negative = myopia",
                            -8.0, 8.0, -3.0, 0.25)
pupil = st.sidebar.slider("Pupil diameter (mm)", 2.0, 6.0, 4.0, 0.5)

st.sidebar.markdown("---")
use_lens = st.sidebar.checkbox("Wear hybrid contact lens", value=True)
use_diffractive = st.sidebar.checkbox("Enable diffractive layer", value=True,
                                      disabled=not use_lens)
material = st.sidebar.selectbox("Lens material",
                                ["silicone-hydrogel", "hydrogel", "PMMA"],
                                disabled=not use_lens)
design_lam = st.sidebar.slider("Diffractive design wavelength (nm)",
                               450, 650, 550, 10,
                               disabled=not use_lens) / 1000.0

mode = st.sidebar.radio("Lens fitting mode",
                        ["Auto-fit", "Optimize", "Manual"],
                        disabled=not use_lens,
                        help="Auto-fit: paraxial base power + LCA-nulling "
                             "diffractive power. Optimize: ray-traced "
                             "polychromatic minimisation. Manual: set powers "
                             "yourself.")
man_base = man_diff = 0.0
opt_a4 = False
if use_lens and mode == "Manual":
    man_base = st.sidebar.slider("Base refractive power (D)", 0.0, 60.0, 35.0, 0.5)
    man_diff = st.sidebar.slider("Diffractive add power (D)", -10.0, 15.0, 4.5, 0.25,
                                 disabled=not use_diffractive)
if use_lens and mode == "Optimize":
    opt_a4 = st.sidebar.checkbox("Also optimise 4th-order term", value=False)

span_um = st.sidebar.slider("Spot-diagram half-width (µm)", 10, 200, 60, 5)


# --------------------------------------------------------------------------
# main panel
# --------------------------------------------------------------------------
st.title("Hybrid Diffractive–Refractive Contact-Lens Simulator")
st.caption("Ray-tracing model of the human eye (Navarro schematic eye) with a "
           "hybrid contact lens for myopia / hyperopia and chromatic-aberration "
           "correction.")

with st.spinner("Tracing rays…"):
    normal, uncorr, corrected, info = get_states(
        error_D, pupil, material, design_lam, use_lens, use_diffractive,
        mode, man_base, man_diff, opt_a4)
    states = [normal, uncorr, corrected]

# headline metrics
g = LAMBDA_GREEN
rms_norm = spot_diagram(normal.system, pupil, g)["rms_um"]
rms_unc = spot_diagram(uncorr.system, pupil, g)["rms_um"]
rms_cor = spot_diagram(corrected.system, pupil, g)["rms_um"]
lca_unc = longitudinal_chromatic_aberration(uncorr.system, LCA_LAMS, g)["lca_D"]
lca_cor = longitudinal_chromatic_aberration(corrected.system, LCA_LAMS, g)["lca_D"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Uncorrected blur (RMS)", f"{rms_unc:.1f} µm")
c2.metric("Corrected blur (RMS)", f"{rms_cor:.1f} µm",
          delta=f"{rms_cor - rms_unc:+.1f} µm", delta_color="inverse")
c3.metric("Uncorrected LCA", f"{lca_unc:.2f} D")
c4.metric("Corrected LCA", f"{lca_cor:.2f} D",
          delta=f"{lca_cor - lca_unc:+.2f} D", delta_color="inverse")

if use_lens and info:
    st.info(f"**Fitted lens** — base refractive power **{info['base']:.2f} D**, "
            f"diffractive add **{info['diff']:.2f} D** at "
            f"{design_lam*1000:.0f} nm design wavelength   "
            f"(front R = {info['R_front']:.2f} mm, back R = {info['R_back']:.2f} mm). "
            "The lens front surface replaces the air–cornea interface, so its "
            "base power includes the corneal front power; the clinically relevant "
            "quantities are the error it corrects and the diffractive add that "
            "flattens the chromatic curve.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Spot diagrams", "MTF", "Chromatic aberration", "About / method"])

with tab1:
    st.pyplot(plot_spots(states, pupil, span_um))
    st.caption("Retinal spot for blue (450 nm), green (550 nm) and red (650 nm) "
               "light, referenced to each spot's centroid. Dashed circle = Airy "
               "(diffraction) radius. Tighter, more overlapped spots = sharper, "
               "more colour-corrected image.")

with tab2:
    st.pyplot(plot_mtf(states, pupil))
    st.caption("Higher MTF = more contrast transmitted at that detail level. "
               "A good correction pushes the green (corrected) curve toward the "
               "normal-eye curve.")

with tab3:
    st.pyplot(plot_lca(uncorr, corrected, use_lens))
    st.caption("Equivalent power vs wavelength. A sloping line means different "
               "colours focus at different depths (chromatic aberration). The "
               "diffractive layer flattens this curve.")

with tab4:
    st.markdown("""
**Model.** Navarro schematic eye (Escudero-Sanz & Navarro 1999): four conic
refracting surfaces with wavelength-dependent (Cauchy) indices, calibrated to
~60 D power and ~2 D of ocular longitudinal chromatic aberration. Ametropia is
introduced by changing axial length (~0.37 mm/D).

**Hybrid contact lens.** A refractive meniscus (silicone-hydrogel / PMMA) whose
front surface carries a diffractive phase profile. Refractive power decreases
with wavelength; diffractive power *increases* with wavelength
(P_d(λ) = P_d0·λ/λ0). Choosing P_d0 to cancel the eye's chromatic slope
flattens the focus-vs-wavelength curve.

**Metrics.** Real 3-D ray tracing (vector Snell + diffractive momentum kick):
- **Spot diagrams** — geometric retinal blur per wavelength.
- **MTF** — from the photopic-weighted ray point-spread function.
- **LCA** — chromatic difference of equivalent power / focus.

These are geometric-optics estimates; at these pupil sizes they capture the
defocus, spherical and chromatic effects the project targets.
""")
