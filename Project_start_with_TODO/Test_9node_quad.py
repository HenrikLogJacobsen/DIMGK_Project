import numpy as np
import quads_with_TODO as quad

# Define the topology (coordinates of the nodes)
ex = np.array([0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5])
ey = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5])

# Element thickness
th = 0.1

# Material properties
E = 2.1e11  # Young's modulus
nu = 0.3    # Poisson's ratio

# Constitutive matrix for plane stress
D = np.array([
    [1.0, nu, 0.],
    [nu, 1.0, 0.],
    [0., 0., (1.0 - nu) / 2.0]]) * E / (1.0 - nu**2)

# Distributed loads (if any)
eq = [1.0, 3.0]

# Initialize stiffness matrix and consistent load vector
Ke = np.zeros((18, 18))
fe = np.zeros((18, 1))

# Calculate stiffness matrix and consistent forces
Ke, fe = quad.quad9_Kmatrix(ex, ey, D, th, eq)

print('Stiffness matrix:\n', Ke)
print('Consistent forces:\n', fe)

# Integration points for quad9 element
numGaussPoints = 3  # Assuming 3x3 Gauss integration for quad9
gp, _ = quad.gauss_points(numGaussPoints)

# Define reference strains for constant strain test
constEx = np.zeros((18,1))
constEy = np.zeros((18,1))
constGamma = np.zeros((18,1))

# Populate 
for i in range(9):
    constEx[i * 2, 0] = ex[i]  # Example values for constant strain
    constEy[i * 2 + 1, 0] = ey[i]
    constGamma[i * 2, 0] = ey[i]
    constGamma[i * 2 + 1, 0] = ex[i]

# Check constant strain at integration points
for xsi in gp:
    for eta in gp:
        B_quad9 = quad.quad9_Bmatrix(xsi, eta, ex, ey)

        # Calculate strains for constant strain state
        Ex_quad9 = B_quad9 @ constEx
        Ey_quad9 = B_quad9 @ constEy
        Gamma_quad9 = B_quad9 @ constGamma

        print('Strains at integration point ({}, {}):'.format(xsi, eta))
        print('Ex:', Ex_quad9)
        print('Ey:', Ey_quad9)
        print('Gamma:', Gamma_quad9)
        print('----------')
