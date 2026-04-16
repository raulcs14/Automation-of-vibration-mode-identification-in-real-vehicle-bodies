"""
Chassis geometry: node coordinates, element connectivity, and subdomain definitions.
Equivalent to buildChassisGeometry.m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChassisGeometry:
    node_coordinates: np.ndarray   # (nNodes, 3) — x, y, z
    element_nodes: np.ndarray      # (nElements, 2) — node indices (0-based)
    subdomains: Dict[str, List[int]]  # zone name → list of node indices


def build_chassis_geometry() -> ChassisGeometry:
    """
    Define the 3D car chassis mesh.

    35 nodes organized in 5 cross-sections along X:
      bumper (x=-1), front (x=0), mid (x=2), rear (x=4), tail (x=5.5)
    Three height levels per section: floor (z=0), belt (z≈1.2), roof (z≈2.4)

    Returns a ChassisGeometry with nodes, elements, and 8 subdomain definitions.
    """
    raise NotImplementedError
