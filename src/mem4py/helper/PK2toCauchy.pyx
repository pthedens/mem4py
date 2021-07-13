# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
cimport cython

from mem4py.helper.area cimport areaSingle
from mem4py.ceygen.ceygenMath cimport dot_mm



cdef extern from "math.h":
    double sqrt(double m)


cdef int PK2toCauchy(double [:] X,
                     double [:] Y,
                     double [:] Z,
                     double J11,
                     double J22,
                     double J12,
                     int [:, ::1] N,
                     unsigned int el,
                     double A0,
                     double [:, :] PK2,
                     double [:] cauchyLocal) except -1:

    """
    Convert second Piola-Kirchhoff stress tensor into Cauchy stress in Voigt form
    
    :param X:           X-coordinates (current configuration)
    :param Y:           Y-coordinates (current configuration)
    :param Z:           Z-coordinates (current configuration)
    :param N:           element connectivity matrix
    :param el:          index of current element
    :param J11:         transformation parameters between initial and current configuration
    :param J12:         transformation parameters between initial and current configuration
    :param J22:         transformation parameters between initial and current configuration
    :param PK2:         second Piola-Kirchhoff stress tensor
    :param cauchy:      Cauchy stress in Voigt form
    
    :return:            void, cauchyLocal changed in memory view
    """

    cdef double j11, j12, j22, detJ, A = 0.

    cdef double [:] e1 = np.zeros(3, dtype=np.double)
    cdef double [:] e2 = np.zeros(3, dtype=np.double)
    cdef double [:] e3 = np.zeros(3, dtype=np.double)

    cdef double [:, :] F = np.zeros((2, 2), dtype=np.double)
    cdef double [:, :] cauchyTensor = np.zeros((2, 2), dtype=np.double)
    cdef double [:, :] temp = np.zeros((2, 2), dtype=np.double)

    # determine components for F
    j11 = sqrt((X[N[el, 2]] - X[N[el, 1]]) * (X[N[el, 2]] - X[N[el, 1]]) +
               (Y[N[el, 2]] - Y[N[el, 1]]) * (Y[N[el, 2]] - Y[N[el, 1]]) +
               (Z[N[el, 2]] - Z[N[el, 1]]) * (Z[N[el, 2]] - Z[N[el, 1]]))
    j12 = ((X[N[el, 2]] - X[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]]) +
           (Y[N[el, 2]] - Y[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]]) +
           (Z[N[el, 2]] - Z[N[el, 1]]) * (Z[N[el, 3]] - Z[N[el, 1]])) / j11

    e1[0] = (X[N[el, 2]] - X[N[el, 1]]) / j11
    e1[1] = (Y[N[el, 2]] - Y[N[el, 1]]) / j11
    e1[2] = (Z[N[el, 2]] - Z[N[el, 1]]) / j11

    areaSingle(X, Y, Z, N[el, :], &A)

    e3[0] = ((Y[N[el, 2]] - Y[N[el, 1]]) * (Z[N[el, 3]] - Z[N[el, 1]]) -
            (Z[N[el, 2]] - Z[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]])) / (2 * A)
    e3[1] = ((Z[N[el, 2]] - Z[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]]) -
            (X[N[el, 2]] - X[N[el, 1]]) * (Z[N[el, 3]] - Z[N[el, 1]])) / (2 * A)
    e3[2] = ((X[N[el, 2]] - X[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]]) -
            (Y[N[el, 2]] - Y[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]])) / (2 * A)

    e2[0] = e3[1] * e1[2] - e3[2] * e1[1]
    e2[1] = e3[2] * e1[0] - e3[0] * e1[2]
    e2[2] = e3[0] * e1[1] - e3[1] * e1[0]

    j22 = (X[N[el, 3]] - X[N[el, 1]]) * e2[0] + \
          (Y[N[el, 3]] - Y[N[el, 1]]) * e2[1] + \
          (Z[N[el, 3]] - Z[N[el, 1]]) * e2[2]

    # deformation gradient tensor F
    F[0, 0] = j11 / J11
    F[0, 1] = (J11 * j12 - j11 * J12) / (J11 * J22)
    F[1, 1] = j22 / J22

    detJ = (j11 * j22) / (J11 * J22)

    dot_mm(PK2, F.T, temp)
    dot_mm(F, temp, cauchyTensor)

    cauchyLocal[0] = cauchyTensor[0, 0]
    cauchyLocal[1] = cauchyTensor[1, 1]
    cauchyLocal[2] = cauchyTensor[0, 1]
    