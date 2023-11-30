import numpy as np
import quads_with_TODO as quad

# Define the topology
ex = np.array([0.0, 1.0, 1.0, 0.0])
ey = np.array([0.0, 0.0, 1.0, 1.0])

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

# Distributed loads
eq = [1.0, 3.0]

# Initialize stiffness matrix and consistent load vector
Ke = np.zeros((8, 8))
fe = np.zeros((8, 1))

# Rigid body motion vectors
rigX = np.zeros((8, 1))
rigY = np.zeros((8, 1))
rigR = np.zeros((8, 1))

# Populate the rigid body motion vectors
for i in range(4):
    rigX[i * 2, 0] = 1.0
    rigY[i * 2 + 1, 0] = 1.0
    rigR[i * 2, 0] = ey[i]
    rigR[i * 2 + 1, 0] = -ex[i]

# Calculate stiffness matrix and consistent forces
Ke, fe = quad.quad4_Kmatrix(ex, ey, D, th, eq)

print('Stiffness matrix:\n', Ke)
print('Consistent forces:\n', fe)

# Calculate forces from rigid body motions
fx = Ke @ rigX
fy = Ke @ rigY
fr = Ke @ rigR

print('Force from rigX translation:\n', fx)
print('Force from rigY translation:\n', fy)
print('Force from rigR rotation:\n', fr)

# Integration points for quad4 element
numGaussPoints = 2  # Number of Gauss points
gp, _ = quad.gauss_points(numGaussPoints)

# Define reference strains for constant strain test
constEx = np.zeros((8,1))
constEy = np.zeros((8,1))
constGamma = np.zeros((8,1))

# Populate 
for i in range(4):
    constEx[i * 2, 0] = ex[i]  # Example values for constant strain
    constEy[i * 2 + 1, 0] = ey[i]
    constGamma[i * 2, 0] = ey[i]
    constGamma[i * 2 + 1, 0] = ex[i]

# Check constant strain at integration points
for xsi in gp:
    for eta in gp:
        B_quad4 = quad.quad4_Bmatrix(xsi, eta, ex, ey)

        # Calculate strains for constant strain state
        Ex_quad4 = B_quad4 @ constEx
        Ey_quad4 = B_quad4 @ constEy
        Gamma_quad4 = B_quad4 @ constGamma

        print('Strains at integration point ({}, {}):'.format(xsi, eta))
        print('Ex:', Ex_quad4)
        print('Ey:', Ey_quad4)
        print('Gamma:', Gamma_quad4)
        print('----------')