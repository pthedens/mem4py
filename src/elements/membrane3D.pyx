import numpy as np
cimport numpy as np
cimport cython

from src.ceygen.math cimport dot_mv
from src.ceygen.math cimport dot_mm
from src.ceygen.math cimport multiply_ms

cdef extern from "math.h":
    double sqrt(double m)
    double cos(double m)
    double sin(double m)
    double atan2(double m, double n)
    
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.profile(True)
cdef void membrane3DStrain(double [:] X,
                           double [:] Y,
                           double [:] Z,
                           int [:, ::1] N,
                           unsigned int el,
                           double J11,
                           double J22,
                           double J12,
                           unsigned int [:] allDofMem,
                           double [:] ELocal):

    cdef:
        g11, g12, g22
        
    # Find degrees of freedom from current element
    allDofMem[0] = 3 * (N[el, 1] + 1) - 3
    allDofMem[1] = 3 * (N[el, 1] + 1) - 2
    allDofMem[2] = 3 * (N[el, 1] + 1) - 1
    allDofMem[3] = 3 * (N[el, 2] + 1) - 3
    allDofMem[4] = 3 * (N[el, 2] + 1) - 2
    allDofMem[5] = 3 * (N[el, 2] + 1) - 1
    allDofMem[6] = 3 * (N[el, 3] + 1) - 3
    allDofMem[7] = 3 * (N[el, 3] + 1) - 2
    allDofMem[8] = 3 * (N[el, 3] + 1) - 1

    # covariant components of the metric tensor in current configuration
    g11 = (X[N[el, 2]] - X[N[el, 1]]) * (X[N[el, 2]] - X[N[el, 1]]) + \
          (Y[N[el, 2]] - Y[N[el, 1]]) * (Y[N[el, 2]] - Y[N[el, 1]]) + \
          (Z[N[el, 2]] - Z[N[el, 1]]) * (Z[N[el, 2]] - Z[N[el, 1]])

    g12 = (X[N[el, 2]] - X[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]]) + \
          (Y[N[el, 2]] - Y[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]]) + \
          (Z[N[el, 2]] - Z[N[el, 1]]) * (Z[N[el, 3]] - Z[N[el, 1]])

    g22 = (X[N[el, 3]] - X[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]]) + \
          (Y[N[el, 3]] - Y[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]]) + \
          (Z[N[el, 3]] - Z[N[el, 1]]) * (Z[N[el, 3]] - Z[N[el, 1]])

    # local strain (Cartesian coordinates), ELocal = Q * ECurv, ECurv = 0.5 * (g_ab - G_ab)
    ELocal[0] = 0.5 * g11 / (J11 * J11) - 0.5
    ELocal[1] = 0.5 * (g22 * J11 -
                       g12 * J12 +
                       g11 * J12 * J12 / J11 -
                       g12 * J12) / (J11 * J22 * J22) - \
                0.5
    ELocal[2] = (g12 - g11 * J12 / J11) / (J11 * J22) / 2
    
    
    
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.profile(True)
cdef void membrane3DStress(double [:, :] Cmat,
                           double [:] strainVoigt,
                           double [:] stressVoigt,
                           double * S1,
                           double * S2,
                           double * theta):
    
    cdef:
        double a, b
        
    # determine stress in fibre coordinate system
    dot_mv(Cmat, strainVoigt, stressVoigt)

    # principal strain direction from elastic strain
    theta[0] = 0.5 * np.arctan2(2 * strainVoigt[2], (strainVoigt[0] - strainVoigt[1]))

    # principal stress
    a = (stressVoigt[0] + stressVoigt[1]) / 2
    b = stressVoigt[2] * stressVoigt[2] - stressVoigt[0] * stressVoigt[1]
    S1[0] = a + sqrt(a * a + b)
    S2[0] = a - sqrt(a * a + b)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.profile(True)
cdef void membrane3DBmat(double [:] X,
                         double [:] Y,
                         double [:] Z,
                         int [:, ::1] N,
                         double J11,
                         double J22,
                         double J12,
                         unsigned int el,
                         double [:] SLocal,
                         double [:] s,
                         double [:, :] BmatLocal):
    
    cdef:
        double [:, :] Q = np.zeros((3, 3), dtype=np.double)
        double [:, :] BmatCurv = np.zeros((3, 9), dtype=np.double)
        
    # fill Q matrix
    Q[0, 0] = 1 / (J11 * J11)

    Q[1, 0] = (J12 * J12) / (J11 * J11 * J22 * J22)
    Q[1, 1] = 1 / (J22 * J22)
    Q[1, 2] = - J12 / (J11 * J22 * J22)

    Q[2, 0] = - 2 * J12 / (J11 * J11 * J22)
    Q[2, 2] = 1 / (J11 * J22)

    # strain displacement matrix in curvilinear coordinate system
    BmatCurv[0, 0] = - (X[N[el, 2]] - X[N[el, 1]])
    BmatCurv[0, 1] = - (Y[N[el, 2]] - Y[N[el, 1]])
    BmatCurv[0, 2] = - (Z[N[el, 2]] - Z[N[el, 1]])

    BmatCurv[0, 3] = X[N[el, 2]] - X[N[el, 1]]
    BmatCurv[0, 4] = Y[N[el, 2]] - Y[N[el, 1]]
    BmatCurv[0, 5] = Z[N[el, 2]] - Z[N[el, 1]]

    BmatCurv[1, 0] = - (X[N[el, 3]] - X[N[el, 1]])
    BmatCurv[1, 1] = - (Y[N[el, 3]] - Y[N[el, 1]])
    BmatCurv[1, 2] = - (Z[N[el, 3]] - Z[N[el, 1]])

    BmatCurv[1, 6] = X[N[el, 3]] - X[N[el, 1]]
    BmatCurv[1, 7] = Y[N[el, 3]] - Y[N[el, 1]]
    BmatCurv[1, 8] = Z[N[el, 3]] - Z[N[el, 1]]

    BmatCurv[2, 0] = 2 * X[N[el, 1]] - X[N[el, 2]] - X[N[el, 3]]
    BmatCurv[2, 1] = 2 * Y[N[el, 1]] - Y[N[el, 2]] - Y[N[el, 3]]
    BmatCurv[2, 2] = 2 * Z[N[el, 1]] - Z[N[el, 2]] - Z[N[el, 3]]

    BmatCurv[2, 3] = X[N[el, 3]] - X[N[el, 1]]
    BmatCurv[2, 4] = Y[N[el, 3]] - Y[N[el, 1]]
    BmatCurv[2, 5] = Z[N[el, 3]] - Z[N[el, 1]]

    BmatCurv[2, 6] = X[N[el, 2]] - X[N[el, 1]]
    BmatCurv[2, 7] = Y[N[el, 2]] - Y[N[el, 1]]
    BmatCurv[2, 8] = Z[N[el, 2]] - Z[N[el, 1]]

    # BLocal = Q * BmatCurv
    dot_mm(Q, BmatCurv, BmatLocal)

    # strain stiffness contribution
    dot_mv(Q.T, SLocal, s)
    
    

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.profile(True)
cdef void membrane3DKmat(double [:] X,
                         double [:] Y,
                         double [:] Z,
                         int [:, ::1] N,
                         double [:, :] BmatLocal,
                         double [:, :] Cmat,
                         double [:, :] KMem,
                         double [:] s,
                         double t,
                         double area,
                         double p,
                         unsigned int el):

    cdef:
        double [:, :] CB = np.empty((3, 9), dtype=np.double)
        
    # Kloc = BFibre.T * C * BFibre
    dot_mm(BmatLocal.T, dot_mm(Cmat, BmatLocal, CB), KMem)

    multiply_ms(KMem, area * t, KMem)

    # external tangent matrix
    KMem[0, 0] += (s[0] + s[1] + 2 * s[2]) * area * t
    KMem[0, 1] -= (Z[N[el, 2]] - Z[N[el, 3]]) * p / 6
    KMem[0, 2] -= (Y[N[el, 3]] - Y[N[el, 2]]) * p / 6
    KMem[0, 3] -= (s[0] + s[2]) * area * t
    KMem[0, 4] -= (Z[N[el, 3]] - Z[N[el, 1]]) * p / 6
    KMem[0, 5] -= (Y[N[el, 1]] - Y[N[el, 3]]) * p / 6
    KMem[0, 6] -= (s[1] + s[2]) * area * t
    KMem[0, 7] -= (Z[N[el, 1]] - Z[N[el, 2]]) * p / 6
    KMem[0, 8] -= (Y[N[el, 2]] - Y[N[el, 1]]) * p / 6

    KMem[1, 0] -= (Z[N[el, 3]] - Z[N[el, 2]]) * p / 6
    KMem[1, 1] += (s[0] + s[1] + 2 * s[2]) * area * t
    KMem[1, 2] -= (X[N[el, 2]] - X[N[el, 3]]) * p / 6
    KMem[1, 3] -= (Z[N[el, 1]] - Z[N[el, 3]]) * p / 6
    KMem[1, 4] -= (s[0] + s[2]) * area * t
    KMem[1, 5] -= (X[N[el, 3]] - X[N[el, 1]]) * p / 6
    KMem[1, 6] -= (Z[N[el, 2]] - Z[N[el, 1]]) * p / 6
    KMem[1, 7] -= (s[1] + s[2]) * area * t
    KMem[1, 8] -= (X[N[el, 1]] - X[N[el, 2]]) * p / 6

    KMem[2, 0] -= (Y[N[el, 2]] - Y[N[el, 3]]) * p / 6
    KMem[2, 1] -= (X[N[el, 3]] - X[N[el, 2]]) * p / 6
    KMem[2, 2] += (s[0] + s[1] + 2 * s[2]) * area * t
    KMem[2, 3] -= (Y[N[el, 3]] - Y[N[el, 1]]) * p / 6
    KMem[2, 4] -= (X[N[el, 1]] - X[N[el, 3]]) * p / 6
    KMem[2, 5] -= (s[0] + s[2]) * area * t
    KMem[2, 6] -= (Y[N[el, 1]] - Y[N[el, 2]]) * p / 6
    KMem[2, 7] -= (X[N[el, 2]] - X[N[el, 1]]) * p / 6
    KMem[2, 8] -= (s[1] + s[2]) * area * t

    KMem[3, 0] -= (s[0] + s[2]) * area * t
    KMem[3, 1] -= (Z[N[el, 2]] - Z[N[el, 3]]) * p / 6
    KMem[3, 2] -= (Y[N[el, 3]] - Y[N[el, 2]]) * p / 6
    KMem[3, 3] += s[0] * area * t
    KMem[3, 4] -= (Z[N[el, 3]] - Z[N[el, 1]]) * p / 6
    KMem[3, 5] -= (Y[N[el, 1]] - Y[N[el, 3]]) * p / 6
    KMem[3, 6] += s[2] * area * t
    KMem[3, 7] -= (Z[N[el, 1]] - Z[N[el, 2]]) * p / 6
    KMem[3, 8] -= (Y[N[el, 2]] - Y[N[el, 1]]) * p / 6

    KMem[4, 0] -= (Z[N[el, 3]] - Z[N[el, 2]]) * p / 6
    KMem[4, 1] -= (s[0] + s[2]) * area * t
    KMem[4, 2] -= (X[N[el, 2]] - X[N[el, 3]]) * p / 6
    KMem[4, 3] -= (Z[N[el, 1]] - Z[N[el, 3]]) * p / 6
    KMem[4, 4] += s[0] * area * t
    KMem[4, 5] -= (X[N[el, 3]] - X[N[el, 1]]) * p / 6
    KMem[4, 6] -= (Z[N[el, 2]] - Z[N[el, 1]]) * p / 6
    KMem[4, 7] += s[2] * area * t
    KMem[4, 8] -= (X[N[el, 1]] - X[N[el, 2]]) * p / 6

    KMem[5, 0] -= (Y[N[el, 2]] - Y[N[el, 3]]) * p / 6
    KMem[5, 1] -= (X[N[el, 3]] - X[N[el, 2]]) * p / 6
    KMem[5, 2] -= (s[0] + s[2]) * area * t
    KMem[5, 3] -= (Y[N[el, 3]] - Y[N[el, 1]]) * p / 6
    KMem[5, 4] -= (X[N[el, 1]] - X[N[el, 3]]) * p / 6
    KMem[5, 5] += s[0] * area * t
    KMem[5, 6] -= (Y[N[el, 1]] - Y[N[el, 2]]) * p / 6
    KMem[5, 7] -= (X[N[el, 2]] - X[N[el, 1]]) * p / 6
    KMem[5, 8] += s[2] * area * t

    KMem[6, 0] -= (s[1] + s[2]) * area * t
    KMem[6, 1] -= (Z[N[el, 2]] - Z[N[el, 3]]) * p / 6
    KMem[6, 2] -= (Y[N[el, 3]] - Y[N[el, 2]]) * p / 6
    KMem[6, 3] += s[2] * area * t
    KMem[6, 4] -= (Z[N[el, 3]] - Z[N[el, 1]]) * p / 6
    KMem[6, 5] -= (Y[N[el, 1]] - Y[N[el, 3]]) * p / 6
    KMem[6, 6] += s[1] * area * t
    KMem[6, 7] -= (Z[N[el, 1]] - Z[N[el, 2]]) * p / 6
    KMem[6, 8] -= (Y[N[el, 2]] - Y[N[el, 1]]) * p / 6

    KMem[7, 0] -= (Z[N[el, 3]] - Z[N[el, 2]]) * p / 6
    KMem[7, 1] -= (s[1] + s[2]) * area * t
    KMem[7, 2] -= (X[N[el, 2]] - X[N[el, 3]]) * p / 6
    KMem[7, 3] -= (Z[N[el, 1]] - Z[N[el, 3]]) * p / 6
    KMem[7, 4] += s[2] * area * t
    KMem[7, 5] -= (X[N[el, 3]] - X[N[el, 1]]) * p / 6
    KMem[7, 6] -= (Z[N[el, 2]] - Z[N[el, 1]]) * p / 6
    KMem[7, 7] += s[1] * area * t
    KMem[7, 8] -= (X[N[el, 1]] - X[N[el, 2]]) * p / 6

    KMem[8, 0] -= (Y[N[el, 2]] - Y[N[el, 3]]) * p / 6
    KMem[8, 1] -= (X[N[el, 3]] - X[N[el, 2]]) * p / 6
    KMem[8, 2] -= (s[1] + s[2]) * area * t
    KMem[8, 3] -= (Y[N[el, 3]] - Y[N[el, 1]]) * p / 6
    KMem[8, 4] -= (X[N[el, 1]] - X[N[el, 3]]) * p / 6
    KMem[8, 5] += s[2] * area * t
    KMem[8, 6] -= (Y[N[el, 1]] - Y[N[el, 2]]) * p / 6
    KMem[8, 7] -= (X[N[el, 2]] - X[N[el, 1]]) * p / 6
    KMem[8, 8] += s[1] * area * t
