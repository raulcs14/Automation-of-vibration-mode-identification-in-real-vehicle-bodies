"""
Static reference model: generates 11 normalized displacement patterns.
Equivalent to Static_Model.m
"""

import numpy as np
from pathlib import Path
from simple_model.geometry.chassis import build_chassis_geometry
from simple_model.fem.stiffness import form_stiffness
from simple_model.fem.mass import form_mass
from simple_model.fem.solver import inertia_relief
import config

DATA_DIR = Path("data/simple_model")

REF_NAMES = [
    "Heave",
    "Pitch",
    "Lateral",
    "Roll/Torsion",
    "Roll front",
    "Roll rear",
    "Roof heave",
    "Pitch roof+floor",
    "Torsion roof+floor",
    "Combo roll+heave",
    "Forced torsion",
]


def run_static_model() -> None:
    """
    Build chassis, apply 11 load cases, solve via inertia relief,
    normalize each solution, and save to data/simple_model/static_reference_moves.npz
    """
    raise NotImplementedError
