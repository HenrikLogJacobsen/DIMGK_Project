# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018

@author: bjohau
"""
import numpy as np

def tri3_area(ex, ey):
    """
    Compute the area of a triangle element

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :return area of the triangle
    """

    tmp = np.matrix([[1, ex[0], ey[0]],
                     [1, ex[1], ey[1]],
                     [1, ex[2], ey[2]]])

    A2 = np.linalg.det(tmp)  # Double of triangle area
    A = A2 / 2.0
    return A


def tri3_Bmatrix(ex, ey):
    """
    Compute the strain displacement matrix for a 3 node triangular membrane element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :return  [3 x 6] strain displacement matrix
    """

    A = tri3_area(ex, ey)
    A2 = 2.0 * A

    cyclic_ijk = [0, 1, 2, 0, 1]  # Cyclic permutation of the nodes i,j,k

    zi_px = np.zeros(3)  # Partial derivative with respect to x
    zi_py = np.zeros(3)  # Partial derivative with respect to y

    for i in range(3):
        j = cyclic_ijk[i + 1]
        k = cyclic_ijk[i + 2]
        zi_px[i] = (ey[j] - ey[k]) / A2
        zi_py[i] = (ex[k] - ex[j]) / A2

    B = np.array([
        [zi_px[0], 0, zi_px[1], 0, zi_px[2], 0],
        [0, zi_py[0], 0, zi_py[1], 0, zi_py[2]],
        [zi_py[0], zi_px[0], zi_py[1], zi_px[1], zi_py[2], zi_px[2]]])

    return B


def tri3_Kmatrix(ex, ey, D, th, eq=None):
    """
    Compute the stiffness matrix for a two dimensional beam element.

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """

    A = tri3_area(ex, ey)
    A2 = 2.0 * A

    cyclic_ijk = [0, 1, 2, 0, 1]  # Cyclic permutation of the nodes i,j,k

    zi_px = np.zeros(3)  # Partial derivative with respect to x
    zi_py = np.zeros(3)  # Partial derivative with respect to y

    for i in range(3):
        j = cyclic_ijk[i + 1]
        k = cyclic_ijk[i + 2]
        zi_px[i] = (ey[j] - ey[k]) / A2
        zi_py[i] = (ex[k] - ex[j]) / A2

    B = tri3_Bmatrix(ex, ey)

    Ke = (B.T @ D @ B) * A * th

    if eq is None:
        return Ke
    else:
        fx = A * th * eq[0] / 3.0
        fy = A * th * eq[1] / 3.0
        fe = np.array([[fx], [fy], [fx], [fy], [fx], [fy]])
        return Ke, fe


def tri3_cornerstresses(ex, ey, D, th, elDispVec):
    """
    Compute the corner stresses for all 3 corner nodes

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """

    B = tri3_Bmatrix(ex, ey)

    strain = B @ elDispVec
    stress = D @ strain
    
    cornerStresses = []
    for inod in range(3):
        cornerStresses.append([stress[0], stress[1], stress[2]])

    return cornerStresses
    
def zeta_partials_x_and_y(ex,ey):
    """
    Compute partials of area coordinates with respect to x and y.
    
    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    """
    tmp = np.array([[1,ex[0],ey[0]],
                    [1,ex[1],ey[1]],
                    [1,ex[2],ey[2]]])
    
    A2 = np.linalg.det(tmp)  # Double of triangle area
       
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k
    
    zeta_px = np.zeros(3)           # Partial derivative with respect to x
    zeta_py = np.zeros(3)           # Partial derivative with respect to y

    for i in range(3):
        j = cyclic_ijk[i+1]
        k = cyclic_ijk[i+2]
        zeta_px[i] = (ey[j] - ey[k]) / A2
        zeta_py[i] = (ex[k] - ex[j]) / A2

    return zeta_px, zeta_py

# Functions for 6 node triangle

def tri6_cornerstresses(ex, ey, D, th, elDispVec):
    """
    Compute the corner stresses for all 3 corner nodes

    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """

    zetaCorner = np.array([[1.0,0.0,0.0],
                        [0.0,1.0,0.0],
                        [0.0,0.0,1.0]])


    cornerStresses = []
    for inod in range(3):
        B = tri6_Bmatrix(zetaCorner[inod], ex, ey)
        strain = B @ elDispVec
        stress = D @ strain
        cornerStresses.append([stress[0], stress[1], stress[2]])

    return cornerStresses
    
def tri6_area(ex,ey):
        
    tmp = np.array([[1,ex[0],ey[0]],
                    [1,ex[1],ey[1]],
                    [1,ex[2],ey[2]]])
    
    A = np.linalg.det(tmp) / 2
    
    return A


def tri6_shape_functions(zeta):
    
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k

    N6 = np.zeros(6)

    for i in range(3):
        j = cyclic_ijk[i+1]
        N6[i] = zeta[i] * 2 * (zeta[i] - 1/2)
        N6[i+3] = 4 * zeta[i] * zeta[j]

    return N6


def tri6_N_matrix(zeta):

    N_mat = np.zeros((2,12))
    N6 = tri6_shape_functions(zeta)
    
    for i in range(6):
        N_mat[0,i*2] = N6[i]
        N_mat[1,i*2+1] = N6[i]

    return N_mat

def tri6_shape_function_partials_x_and_y(zeta,ex,ey):
    
    zeta_px, zeta_py = zeta_partials_x_and_y(ex,ey)
    
    N6_px = np.zeros(6)
    N6_py = np.zeros(6)
    
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k

    for i in range(3):
        j = cyclic_ijk[i+1]
        N6_px[i] = 4 * zeta_px[i] * zeta[i] - zeta_px[i]
        N6_py[i] = 4 * zeta_py[i] * zeta[i] - zeta_py[i]
        N6_px[i+3] = 4 * (zeta[j]*zeta_px[i] + zeta[i]*zeta_px[j])
        N6_py[i+3] = 4 * (zeta[j]*zeta_py[i] + zeta[i]*zeta_py[j])

    return N6_px, N6_py


def tri6_Bmatrix(zeta,ex,ey):
    
    nx,ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)

    Bmatrix = np.zeros((3,12))

    Bmatrix[0,0::2] = nx
    Bmatrix[1,1::2] = ny
    Bmatrix[2,0::2] = ny
    Bmatrix[2,1::2] = nx

    return Bmatrix


def tri6_Kmatrix(ex,ey,D,th,eq=None):
    
    zetaInt = np.array([[0.5,0.5,0.0],
                        [0.0,0.5,0.5],
                        [0.5,0.0,0.5]])
    
    wInt = np.array([1.0/3.0,1.0/3.0,1.0/3.0])

    A    = tri6_area(ex,ey)

    Ke = np.zeros((12,12))

    for i in range(3):
        for j in range(3):
            zeta = zetaInt[i,:]
            wi = wInt[i]
            wj = wInt[j]
            B = tri6_Bmatrix(zeta, ex, ey)
            Ke += wi * wj * ((B.T @ D) @ B) * A * th
    if eq is None:
        return Ke
    else:
        fe = np.zeros((12,1))
        eqMat = np.array([[eq[0]],[eq[1]]])
        for i in range(3):
            for j in range(3):
                zeta = zetaInt[i,:]
                wi = wInt[i]
                wj = wInt[j]
                N = tri6_N_matrix(zeta)                
                fe += N.T @ eqMat * wi * wj * A * th

            # ---------------------------- #

        return Ke, fe







  
