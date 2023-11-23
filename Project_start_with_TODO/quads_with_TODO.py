# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np
import sys

def gauss_points(iRule):
    """
    Returns gauss coordinates and weight given integration number

    Parameters:

        iRule = number of integration points

    Returns:

        gp : row-vector containing gauss coordinates
        gw : row-vector containing gauss weight for integration point

    """
    gauss_position = [[ 0.000000000],
                      [-0.577350269,  0.577350269],
                      [-0.774596669,  0.000000000,  0.774596669],
                      [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
                      [-0.9061798459, -0.5384693101, 0.0000000000, 0.5384693101, 0.9061798459]]
    gauss_weight   = [[2.000000000],
                      [1.000000000,   1.000000000],
                      [0.555555556,   0.888888889,  0.555555556],
                      [0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451],
                      [0.2369268850,  0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]]


    if iRule < 1 and iRule > 5:
        sys.exit("Invalid number of integration points.")

    idx = iRule - 1
    return gauss_position[idx], gauss_weight[idx]


# --------------------- 4 node quad ---------------------


def quad4_shapefuncs(xsi, eta):
    """
    Calculates shape functions evaluated at xi, eta
    """
    # ----- Shape functions -----
    N = np.zeros(4)
    N[0] = 0.25 * (1 - xsi) * (1 - eta)
    N[1] = 0.25 * (1 + xsi) * (1 - eta)
    N[2] = 0.25 * (1 + xsi) * (1 + eta)
    N[3] = 0.25 * (1 - xsi) * (1 + eta)
    return N

def quad4_shapefuncs_grad_xsi(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. xsi
    """
    # ----- Derivatives of shape functions with respect to xsi -----
    Ndxi = np.zeros(4)
    Ndxi[0] = -0.25 * (1 - eta)
    Ndxi[1] =  0.25 * (1 - eta)
    Ndxi[2] =  0.25 * (1 + eta)
    Ndxi[3] = -0.25 * (1 + eta)
    return Ndxi


def quad4_shapefuncs_grad_eta(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. eta
    """
    # ----- Derivatives of shape functions with respect to eta -----
    Ndeta = np.zeros(4)
    Ndeta[0] = -0.25 * (1 - xsi)
    Ndeta[1] = -0.25 * (1 + xsi)
    Ndeta[2] =  0.25 * (1 + xsi)
    Ndeta[3] =  0.25 * (1 - xsi)
    return Ndeta

#Looks correct in Paraview except for Stress in Z-direction
def quad4_cornerstresses(ex, ey, D, thickness, elDisp):
    cornerNodes = np.array([[-1.0,-1.0],
                           [ 1.0,-1.0],
                           [ 1.0, 1.0],
                           [-1.0, 1.0]])

    cornerStresses = []
    for inode in range(4):
        B = quad4_Bmatrix(cornerNodes[inode, 0], cornerNodes[inode, 1], ex, ey)
        strain = B @ elDisp
        stress = D @ strain
        cornerStresses.append([stress[0], stress[1], stress[2]])

    return cornerStresses


def quad4_Bmatrix(xsi, eta, ex, ey):
    """
    Calculates the B matrix for a 4 node quadrilateral element
    """
    # ----- Derivatives of shape functions -----
    Ndxi = quad4_shapefuncs_grad_xsi(xsi, eta)
    Ndeta = quad4_shapefuncs_grad_eta(xsi, eta)

    # ----- Jacobian matrix -----
    H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
    G = np.array([Ndxi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta
    J = G @ H

    # ----- Inverse of Jacobian matrix -----
    invJ = np.linalg.inv(J)

    # ----- Derivatives of shape functions with respect to x and y -----
    Ndx = invJ[0, 0] * Ndxi + invJ[0, 1] * Ndeta
    Ndy = invJ[1, 0] * Ndxi + invJ[1, 1] * Ndeta

    # ----- B matrix -----
    B = np.zeros((3, 8))
    B[0, 0::2] = Ndx
    B[1, 1::2] = Ndy
    B[2, 0::2] = Ndy
    B[2, 1::2] = Ndx

    return B


def quad4_Kmatrix(ex, ey, D, thickness, eq=None):
    """
    Compute the stiffness matrix for a four node membrane element.

    Parameters:

        ex  = [x1 ... x4]           Element coordinates. Row matrix
        ey  = [y1 ... y4]
        D   =           Constitutive matrix
        thickness:      Element thickness
        eq = [bx; by]       bx:     body force in x direction
                            by:     body force in y direction

    Returns:

        Ke : element stiffness matrix (8 x 8)
        fe : equivalent nodal forces (4 x 1)

    """
    t = thickness

    if eq is None:
        f = np.zeros((2,1))  # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    Ke = np.zeros((8,8))        # Create zero matrix for stiffness matrix
    fe = np.zeros((8,1))        # Create zero matrix for distributed load

    numGaussPoints = 2  # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):  # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            Ndxsi = quad4_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad4_shapefuncs_grad_eta(xsi, eta)
            N1    = quad4_shapefuncs(xsi, eta)  # Collect shape functions evaluated at xi and eta

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta

            J = G @ H
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            # Strain displacement matrix calculated at position xsi, eta
            B  = quad4_Bmatrix(xsi, eta, ex, ey)

            # Displacement interpolation xsi and eta
            N2 = np.zeros((2,8))
            N2[0, 0::2] = N1
            N2[1, 1::2] = N1

            # Evaluate integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * t * gw[iGauss] * gw[jGauss]

    return Ke, fe  # Returns stiffness matrix and nodal force vector

# --------------------------- 9 node quad ---------------------------

def quad9_shapefuncs(xsi, eta):
    """
    Calculates shape functions evaluated at xi, eta
    """
    # ----- Shape functions -----
    N = np.zeros(9)
    N[0] = 0.25 * xsi * eta * (1-xsi) * (1-eta)
    N[1] = -0.25 * xsi * eta * (1+xsi) * (1-eta)
    N[2] = 0.25 * xsi * eta * (1+xsi) * (1+eta)
    N[3] = -0.25 * xsi * eta * (1-xsi) * (1+eta)
    N[4] = -0.5 * (1+xsi) * (1-xsi) * eta * (1-eta)
    N[5] = 0.5 * xsi * (1+xsi) * (1-eta) * (1+eta)
    N[6] = 0.5 * (1+xsi) * (1-xsi) * eta * (1+eta)
    N[7] = -0.5 * xsi * (1-xsi) * (1-eta) * (1 + eta)
    N[8] = (1+xsi) * (1-xsi) * (1+eta) * (1-eta)
    return N

def quad9_shapefuncs_grad_xsi(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. xsi
    """
    Ndxi = np.zeros(9)
    Ndxi[0] = 0.25 * (1-2*xsi) * eta * (1-eta)
    Ndxi[1] = -0.25 * (1+2*xsi) * eta * (1-eta)
    Ndxi[2] = 0.25 * (1+2*xsi) * eta * (1+eta)
    Ndxi[3] = -0.25 * (1-2*xsi) * eta * (1+eta)
    Ndxi[4] = xsi * eta * (1-eta)
    Ndxi[5] = 0.5 * (1+2*xsi) * (1-eta**2)
    Ndxi[6] = -xsi * eta * (1+eta)
    Ndxi[7] = -0.5 * (1-2*xsi) * (1-eta) * (1 + eta)
    Ndxi[8] = -2 * xsi * (1 - eta**2)
    return Ndxi

def quad9_shapefuncs_grad_eta(xsi, eta):
    """
    Calculates derivatives of shape functions wrt. eta
    """
    Ndeta = np.zeros(9)
    Ndeta[0] = 0.25 * xsi * (1-xsi) * (1-2*eta)
    Ndeta[1] = -0.25 * xsi * (1+xsi) * (1-2*eta)
    Ndeta[2] = 0.25 * xsi * (1+xsi) * (1+2*eta)
    Ndeta[3] = -0.25 * xsi * (1-xsi) * (1+2*eta)
    Ndeta[4] = -0.5 * (1+xsi) * (1-xsi) * (1-2*eta)
    Ndeta[5] = 0.5 * xsi * (1+xsi) * (-2*eta)
    Ndeta[6] = 0.5 * (1+xsi) * (1-xsi) * (1+2*eta)
    Ndeta[7] = -0.5 * xsi * (1-xsi) * (-2*eta)
    Ndeta[8] = (1-xsi**2) * (-2*eta)
    return Ndeta

def quad9_Bmatrix(xsi, eta, ex, ey):
    """
    Calculates the B matrix for a 9 node quadrilateral element
    """
    # ----- Derivatives of shape functions -----
    Ndxi = quad9_shapefuncs_grad_xsi(xsi, eta)
    Ndeta = quad9_shapefuncs_grad_eta(xsi, eta)

    # ----- Jacobian matrix -----
    H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
    G = np.array([Ndxi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta
    J = G @ H

    # ----- Inverse of Jacobian matrix -----
    if (np.linalg.det(J) == 0):
        invJ = np.zeros((2,2))
    else:
        invJ = np.linalg.inv(J)

    # ----- Derivatives of shape functions with respect to x and y -----
    Ndx = invJ[0, 0] * Ndxi + invJ[0, 1] * Ndeta
    Ndy = invJ[1, 0] * Ndxi + invJ[1, 1] * Ndeta

    # ----- B matrix -----
    B = np.zeros((3, 18))
    B[0, 0::2] = Ndx
    B[1, 1::2] = Ndy
    B[2, 0::2] = Ndy
    B[2, 1::2] = Ndx

    return B

def quad9_cornerstresses(ex, ey, D, th, elDisp):
    cornerNodes = np.array([[-1.0,-1.0],
                           [ 1.0,-1.0],
                           [ 1.0, 1.0],
                           [-1.0, 1.0]])

    cornerStresses = []
    for inode in range(4):
        B = quad9_Bmatrix(cornerNodes[inode, 0], cornerNodes[inode, 1], ex, ey)
        strain = B @ elDisp
        stress = D @ strain
        cornerStresses.append([stress[0], stress[1], stress[2]])

    return cornerStresses

def quad9_Kmatrix(ex, ey, D, th, eq=None):
    """
    Calculates the stiffness matrix for a 9 node isoparametric element in plane stress

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """
    if eq is None:
        f = np.zeros((2,1))
    else:
        f = np.array([eq]).T

    Ke = np.zeros((18,18))
    #Ke = np.eye(18) * 1.0e6
    fe = np.zeros((18,1))

    numGaussPoints = 3
    gp, gw = gauss_points(numGaussPoints)

    for iGauss in range(numGaussPoints):
        for jGauss in range(numGaussPoints):
            
            xsi = gp[iGauss]
            eta = gp[jGauss]

            N1    = quad9_shapefuncs(xsi, eta)
            Ndxsi = quad9_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad9_shapefuncs_grad_eta(xsi, eta)

            H = np.transpose([ex, ey])
            G = np.array([Ndxsi, Ndeta])
            J = G @ H
            detJ = np.linalg.det(J)

            B  = quad9_Bmatrix(xsi, eta, ex, ey)

            N2 = np.zeros((2,18))
            N2[0, 0::2] = N1
            N2[1, 1::2] = N1

            Ke += (B.T) @ D @ B * detJ * th * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * th * gw[iGauss] * gw[jGauss]

    return Ke, fe





  
