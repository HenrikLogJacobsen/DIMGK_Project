import numpy as np
import quads_with_TODO as quad

# Define the topology (coordinates of the nodes)
# For a 9-node quadrilateral, define the corner nodes first and then mid-side nodes
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

# Rigid body motion vectors
rigX = np.zeros((18, 1))
rigY = np.zeros((18, 1))
rigR = np.zeros((18, 1))

# Populate the rigid body motion vectors
for i in range(9):
    rigX[i * 2, 0] = 1.0
    rigY[i * 2 + 1, 0] = 1.0
    rigR[i * 2, 0] = ey[i]
    rigR[i * 2 + 1, 0] = -ex[i]

# Calculate stiffness matrix and consistent forces
Ke, fe = quad.quad9_Kmatrix(ex, ey, D, th, eq)

print('Stiffness matrix:\n', Ke)
print('Consistent forces:\n', fe)

# Calculate forces from rigid body motions
fx = Ke @ rigX
fy = Ke @ rigY
fr = Ke @ rigR

print('Force from rigX translation:\n', fx)
print('Force from rigY translation:\n', fy)
print('Force from rigR rotation:\n', fr)
