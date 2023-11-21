import numpy as np
import quads_with_TODO as quad

# Define the topology (coordinates of the nodes)
# For a 4-node quadrilateral, we typically define the nodes in a counterclockwise manner
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

# Distributed loads (if any)
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
