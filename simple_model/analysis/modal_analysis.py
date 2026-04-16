"""
Modal analysis: computes free-free dynamic modes of the chassis.
Equivalent to Modes_Calculator.m
"""

import numpy as np
from pathlib import Path
from simple_model.geometry.chassis import build_chassis_geometry
from simple_model.fem.stiffness import form_stiffness
from simple_model.fem.mass import form_mass
from simple_model.fem.solver import solve_eigenvalue
import config

DATA_DIR = Path("data/simple_model")

N_RIGID_BODY_MODES = 6
N_ELASTIC_MODES = 30


def build_rigid_body_basis(node_coordinates: np.ndarray) -> np.ndarray:
    """
    Build the rigid-body mode basis R (GDof x 6).
    6 DOFs per node: 3 translations + 3 rigid rotations about the centroid.
    """
    raise NotImplementedError


def run_modal_analysis() -> None:
    """
    Solve K φ = λ M φ (free-free), skip modes 1-6 (rigid body),
    keep modes 7-36 (30 elastic modes), and save to
    data/simple_model/dynamic_modes.npz
    """
    raise NotImplementedError
