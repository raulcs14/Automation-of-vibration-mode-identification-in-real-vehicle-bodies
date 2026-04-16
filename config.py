"""
Global material and geometric parameters shared across models.
"""

# Material properties (steel)
E = 210e9       # Young's modulus [Pa]
G = 84e9        # Shear modulus [Pa]
RHO = 7850.0    # Density [kg/m³]

# Cross-section properties
A = 0.02        # Cross-sectional area [m²]
IY = 10e-5      # Second moment of area about y [m⁴]
IZ = 20e-5      # Second moment of area about z [m⁴]
J = 5e-5        # Torsional constant [m⁴]

# DOFs per node
NDOF = 6
