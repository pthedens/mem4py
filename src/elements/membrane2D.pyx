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
cdef void membrane2DStrain(double [:] X,
                           double [:] Y,
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
    allDofMem[0] = 2 * (N[el, 1] + 1) - 2
    allDofMem[1] = 2 * (N[el, 1] + 1) - 1
    allDofMem[2] = 2 * (N[el, 2] + 1) - 2
    allDofMem[3] = 2 * (N[el, 2] + 1) - 1
    allDofMem[4] = 2 * (N[el, 3] + 1) - 2
    allDofMem[5] = 2 * (N[el, 3] + 1) - 1

    # covariant components of the metric tensor in current configuration
    g11 = (X[N[el, 2]] - X[N[el, 1]]) * (X[N[el, 2]] - X[N[el, 1]]) + \
          (Y[N[el, 2]] - Y[N[el, 1]]) * (Y[N[el, 2]] - Y[N[el, 1]])

    g12 = (X[N[el, 2]] - X[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]]) + \
          (Y[N[el, 2]] - Y[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]])

    g22 = (X[N[el, 3]] - X[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]]) + \
          (Y[N[el, 3]] - Y[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]])

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
cdef void membrane2DStress(double [:, :] Cmat,
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
cdef void membrane2DBmat(double [:] X,
                         double [:] Y,
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
        double [:, :] BmatCurv = np.zeros((3, 6), dtype=np.double)
        
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

    BmatCurv[0, 2] = X[N[el, 2]] - X[N[el, 1]]
    BmatCurv[0, 3] = Y[N[el, 2]] - Y[N[el, 1]]

    BmatCurv[1, 0] = - (X[N[el, 3]] - X[N[el, 1]])
    BmatCurv[1, 1] = - (Y[N[el, 3]] - Y[N[el, 1]])

    BmatCurv[1, 4] = X[N[el, 3]] - X[N[el, 1]]
    BmatCurv[1, 5] = Y[N[el, 3]] - Y[N[el, 1]]

    BmatCurv[2, 0] = 2 * X[N[el, 1]] - X[N[el, 2]] - X[N[el, 3]]
    BmatCurv[2, 1] = 2 * Y[N[el, 1]] - Y[N[el, 2]] - Y[N[el, 3]]

    BmatCurv[2, 2] = X[N[el, 3]] - X[N[el, 1]]
    BmatCurv[2, 3] = Y[N[el, 3]] - Y[N[el, 1]]

    BmatCurv[2, 4] = X[N[el, 2]] - X[N[el, 1]]
    BmatCurv[2, 5] = Y[N[el, 2]] - Y[N[el, 1]]

    # BLocal = Q * BmatCurv
    dot_mm(Q, BmatCurv, BmatLocal)

    # strain stiffness contribution
    dot_mv(Q.T, SLocal, s)
    
    

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void membrane2DKmat(double [:, :] BmatLocal,
                         double [:, :] Cmat,
                         double [:, :] KMem,
                         double [:] s,
                         double t,
                         double area):

    cdef:
        double [:, :] CB = np.empty((3, 6), dtype=np.double)
        
    # Kloc = BFibre.T * C * BFibre
    dot_mm(BmatLocal.T, dot_mm(Cmat, BmatLocal, CB), KMem)

    # external tangent matrix
    KMem[0, 0] += s[0] + s[1] + 2 * s[2]
    KMem[0, 2] -= s[0] + s[2]
    KMem[0, 4] -= s[1] + s[2]

    KMem[1, 1] += s[0] + s[1] + 2 * s[2]
    KMem[1, 3] -= s[0] + s[2]
    KMem[1, 5] -= s[1] + s[2]

    KMem[2, 0] -= s[0] + s[2]
    KMem[2, 2] += s[0]
    KMem[2, 4] += s[2]

    KMem[3, 1] -= s[0] + s[2]
    KMem[3, 3] += s[0]
    KMem[3, 5] += s[2]

    KMem[4, 0] -= s[1] + s[2]
    KMem[4, 2] += s[2]
    KMem[4, 4] += s[1]

    KMem[5, 1] -= s[1] + s[2]
    KMem[5, 3] += s[2]
    KMem[5, 5] += s[1]

    # integrate over volume
    multiply_ms(KMem, area * t, KMem)
