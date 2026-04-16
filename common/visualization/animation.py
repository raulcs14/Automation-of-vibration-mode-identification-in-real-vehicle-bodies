"""
Mode shape animation.
Equivalent to ModesAnimation.m
"""

import numpy as np
import matplotlib.pyplot as plt
from common.visualization.mesh import draw_mesh


def animate_mode(node_coordinates: np.ndarray, element_nodes: np.ndarray,
                 mode: np.ndarray, frequency: float,
                 mode_number: int = 1,
                 scale: float = 50.0,
                 n_frames: int = 200,
                 pause: float = 0.02) -> None:
    """
    Animate a single mode shape with sinusoidal phase.

    Args:
        node_coordinates: (nNodes, 3)
        element_nodes: (nElements, 2)
        mode: (GDof,) mode shape vector
        frequency: Mode frequency in Hz (for title)
        mode_number: Display index (for title)
        scale: Displacement amplification factor
        n_frames: Total animation frames
        pause: Pause between frames in seconds
    """
    raise NotImplementedError
