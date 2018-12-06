import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double sqrt(double m)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void bar2DStrainGreen(double l,
                           double L0,
                           double * strainBar):

    strainBar[0] = (l * l - L0 * L0) / (2 * L0 * L0)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void bar2DCauchyStress(double [:] X,
                            double [:] Y,
                            int [:, ::1] N,
                            double L0,
                            double EBar,
                            double * strainBar,
                            double * stressBar,
                            unsigned int el):

    cdef double X21, Y21, L

    # Nodal distances in xyz system
    X21 = X[N[el, 2]] - X[N[el, 1]]
    Y21 = Y[N[el, 2]] - Y[N[el, 1]]

    # Compute bar length in current configuration
    L = sqrt(X21 * X21 +
             Y21 * Y21)

    strainBar[0] = (L * L - L0 * L0) / (2 * L0 * L0)

    if strainBar[0] < 0:
        stressBar[0] = 0
    else:
        stressBar[0] = EBar * strainBar[0] * L / L0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void bar2DFintAndK(double [:] X,
                        double [:] Y,
                        int [:, ::1] N,
                        double L0,
                        unsigned int el,
                        double areaBar,
                        double E,
                        double [:] Fint,
                        unsigned int [:] allDofBar,
                        double [:, ::1] Klocal):

    cdef:
        double strainBar, L, X21, Y21, f

    # Find degrees of freedom from current element
    allDofBar[0] = 2 * (N[el, 1] + 1) - 2
    allDofBar[1] = 2 * (N[el, 1] + 1) - 1
    allDofBar[2] = 2 * (N[el, 2] + 1) - 2
    allDofBar[3] = 2 * (N[el, 2] + 1) - 1

    # Nodal distances in xy system
    X21 = X[N[el, 2]] - X[N[el, 1]]
    Y21 = Y[N[el, 2]] - Y[N[el, 1]]

    # Compute bar length in current configuration
    L = sqrt(X21 * X21 +
             Y21 * Y21)

    bar2DStrainGreen(L, L0, &strainBar)

    # Compression criterion
    if strainBar < 0:
        Fint[...] = 0
        strainBar = 0
    else:
        # Internal force vector
        f = E * strainBar * areaBar / L0

        Fint[0] = - X21 * f
        Fint[1] = - Y21 * f
        Fint[2] = X21 * f
        Fint[3] = Y21 * f

    # Tangent stiffness matrix
    f = E * areaBar / L0
    Klocal[0, 0] = f * (X21 * X21 + strainBar)
    Klocal[0, 1] = f * X21 * Y21
    Klocal[0, 2] = - f * (X21 * X21 + strainBar)
    Klocal[0, 3] = - f * X21 * Y21

    Klocal[1, 0] = f * Y21 * X21
    Klocal[1, 1] = f * (Y21 * Y21 + strainBar)
    Klocal[1, 2] = - f * Y21 * X21
    Klocal[1, 3] = - f * (Y21 * Y21 + strainBar)

    Klocal[2, 0] = - f * (X21 * X21 + strainBar)
    Klocal[2, 1] = - f * X21 * Y21
    Klocal[2, 2] = f * (X21 * X21 + strainBar)
    Klocal[2, 3] = f * X21 * Y21

    Klocal[3, 0] = - f * Y21 * X21
    Klocal[3, 1] = - f * (Y21 * Y21 + strainBar)
    Klocal[3, 2] = f * Y21 * X21
    Klocal[3, 3] = f * (Y21 * Y21 + strainBar)