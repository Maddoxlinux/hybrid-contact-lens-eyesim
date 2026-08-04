"""
eyesim
======
Ray-tracing model of the human eye plus a hybrid refractive-diffractive
contact lens, for the study of myopia / hyperopia correction and the
reduction of ocular chromatic aberration.

Modules
-------
dispersion : wavelength-dependent refractive indices of the ocular media
surfaces   : optical surface primitives (conic refractive + diffractive)
raytrace   : real 3-D sequential ray tracer and a paraxial engine
eye        : builder for the Navarro schematic eye (parametric ametropia)
lens       : builder for the hybrid refractive-diffractive contact lens
metrics    : spot diagrams, geometric MTF, longitudinal chromatic aberration
optimize   : numerical optimisation of the hybrid lens

All lengths are in millimetres and all wavelengths in micrometres unless a
name says otherwise.  Optical powers are reported in dioptres (1/m).
"""

from .dispersion import OcularMedia, cauchy_index, PHOTOPIC
from .surfaces import Surface, ConicSurface, DiffractiveSurface, ImagePlane
from .raytrace import System, trace_rays, paraxial_power
from .eye import build_navarro_eye, EyeModel
from .lens import build_hybrid_lens, ContactLens, auto_fit_lens
from .metrics import spot_diagram, geometric_mtf, longitudinal_chromatic_aberration

__all__ = [
    "OcularMedia", "cauchy_index", "PHOTOPIC",
    "Surface", "ConicSurface", "DiffractiveSurface", "ImagePlane",
    "System", "trace_rays", "paraxial_power",
    "build_navarro_eye", "EyeModel",
    "build_hybrid_lens", "ContactLens", "auto_fit_lens",
    "spot_diagram", "geometric_mtf", "longitudinal_chromatic_aberration",
]

__version__ = "0.1.0"
