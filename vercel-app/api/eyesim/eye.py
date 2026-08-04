"""
Builder for the Navarro schematic eye, with parametric ametropia.

The four refracting surfaces (anterior/posterior cornea, anterior/posterior
crystalline lens) use the relaxed-accommodation parameters of the Navarro
model (Escudero-Sanz & Navarro 1999).  The retina is placed automatically at
the paraxial image plane of the *nominal* eye at the reference wavelength, so
that the nominal eye is emmetropic by construction.

Ametropia is introduced by moving the retina axially: a longer eye (retina
behind the image) is myopic, a shorter eye is hyperopic.  The axial shift for
a target refractive error E (dioptres) is

    delta_z = -E * n_vitreous / F_eye**2         (metres, then -> mm)

with F_eye the equivalent power of the eye, i.e. the standard relation of
~0.37 mm of axial length per dioptre (~2.7 D/mm).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .dispersion import OcularMedia, LAMBDA_GREEN
from .surfaces import ConicSurface, ImagePlane
from .raytrace import System, paraxial_power


# Navarro relaxed eye: (radius mm, conic k, thickness-to-next mm, medium)
_NAVARRO = [
    # anterior cornea
    dict(R=7.72,  k=-0.26,    t=0.55,  medium=OcularMedia.CORNEA,  sd=5.5),
    # posterior cornea
    dict(R=6.50,  k=0.0,      t=3.05,  medium=OcularMedia.AQUEOUS, sd=5.5),
    # anterior lens
    dict(R=10.20, k=-3.1316,  t=4.00,  medium=OcularMedia.LENS,    sd=4.5),
    # posterior lens
    dict(R=-6.00, k=-1.00,    t=16.3203, medium=OcularMedia.VITREOUS, sd=4.5),
]

# nominal axial length (cornea vertex -> retina), mm
NOMINAL_AXIAL_LENGTH = sum(s["t"] for s in _NAVARRO)


@dataclass
class EyeModel:
    system: System
    axial_length: float
    refractive_error_D: float
    pupil_diameter: float
    retina_z: float


def _build_surfaces(pupil_semi: float, extra_front: list | None = None):
    """Assemble the cornea+lens surfaces starting at z = 0 (cornea vertex)."""
    surfaces = []
    z = 0.0
    for i, s in enumerate(_NAVARRO):
        # the anterior lens surface acts as the aperture stop (iris)
        sd = pupil_semi if i == 2 else s["sd"]
        surfaces.append(ConicSurface(z=z, material_after=s["medium"],
                                     semi_diameter=max(sd, pupil_semi),
                                     R=s["R"], k=s["k"],
                                     name=f"navarro-{i}"))
        z += s["t"]
    return surfaces, z


def build_navarro_eye(refractive_error_D: float = 0.0,
                      pupil_diameter: float = 4.0,
                      lam_ref_um: float = LAMBDA_GREEN,
                      front_optics: list | None = None) -> EyeModel:
    """Return an `EyeModel`.

    Parameters
    ----------
    refractive_error_D : target spherical error at the reference wavelength.
                         Negative = myopia, positive = hyperopia.
    pupil_diameter     : entrance pupil diameter (mm).
    front_optics       : optional list of surfaces (e.g. a contact lens) to
                         place *in front of* the cornea.  Their z-positions
                         must already be negative (upstream of z = 0).
    """
    pupil_semi = pupil_diameter / 2.0
    eye_surfaces, z_after_lens = _build_surfaces(pupil_semi)

    surfaces = []
    if front_optics:
        surfaces.extend(front_optics)
    surfaces.extend(eye_surfaces)

    # locate the emmetropic retina: paraxial image plane of the bare eye
    bare = System([*eye_surfaces, ImagePlane(z=z_after_lens,
                                             material_after=OcularMedia.VITREOUS)])
    power_D, bfd = paraxial_power(bare, lam_ref_um)
    # image plane measured from the last refracting surface (posterior lens)
    z_post_lens = eye_surfaces[-1].z
    retina_emmetropic = z_post_lens + bfd

    # axial shift for the requested refractive error
    n_vit = OcularMedia.VITREOUS.index(lam_ref_um)
    F = power_D / 1000.0                     # 1/mm
    if F == 0:
        delta = 0.0
    else:
        # metres form: dz = -E * n / F_m^2 ; convert to mm
        F_m = power_D                        # dioptres = 1/m
        delta = -refractive_error_D * n_vit / (F_m ** 2) * 1000.0
    retina_z = retina_emmetropic + delta

    surfaces.append(ImagePlane(z=retina_z, material_after=OcularMedia.VITREOUS,
                               semi_diameter=6.0, name="retina"))

    system = System(surfaces, material_before=OcularMedia.AIR)
    axial_length = retina_z - 0.0            # cornea vertex is at z = 0
    return EyeModel(system=system, axial_length=axial_length,
                    refractive_error_D=refractive_error_D,
                    pupil_diameter=pupil_diameter, retina_z=retina_z)
