# import numpy as np
# cimport numpy as np
cimport cython
cdef extern from "math.h":
    double sin(double m)
    double cos(double m)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void CmatStVenantIsotropic(double E,
                                double poisson,
                                double [:, :] Cmat):

    cdef double m, n, C11, C12, C22, C33

    # transform to local coordinates
    # m = cos(theta)
    # n = sin(theta)

    # local constitutive matrix parameters
    C11 = E / (1 - poisson * poisson)
    C12 = poisson * E / (1 - poisson * poisson)
    C22 = E / (1 - poisson * poisson)
    C33 = 0.5 * (1 - poisson) * E / (1 - poisson * poisson)

    # constitutive matrix in global coordinates
    # FIRST ROW
    # Cmat[0, 0] = m * m * (C11 * m * m + C12 * n * n) + \
    #              n * n * (C12 * m * m + C22 * n * n) + \
    #              4 * n * n * m * m * C33
    # Cmat[0, 1] = m * m * (C11 * n * n + C12 * m * m) + \
    #              n * n * (C12 * n * n + C22 * m * m) - \
    #              4 * n * n * m * m * C33
    # Cmat[0, 2] = m * m * m * n * (C11 - C12) + \
    #              m * n * n * n * (C12 - C22) - \
    #              2 * m * n * C33 * (m * m - n * n)
    # # SECOND ROW
    # Cmat[1, 0] = n * n * (C11 * m * m + C12 * n * n) + \
    #              m * m * (C12 * m * m + C22 * n * n) - \
    #              4 * n * n * m * m * C33
    # Cmat[1, 1] = n * n * (C11 * n * n + C12 * m * m) + \
    #              m * m * (C12 * n * n + C22 * m * m) + \
    #              4 * n * n * m * m * C33
    # Cmat[1, 2] = m * m * m * n * (C11 - C12) + \
    #              m * n * n * n * (C12 - C22) - \
    #              2 * m * n * C33 * (m * m - n * n)
    # # THIRD ROW
    # Cmat[2, 0] = m * n * (C11 * m * m + C12 * n * n) - \
    #              m * n * (C12 * m * m + C22 * n * n) - \
    #              (m * m - n * n) * 2 * n * m * C33
    # Cmat[2, 1] = m * n * (C11 * n * n + C12 * m * m) - \
    #              m * n * (C12 * n * n + C22 * m * m) + \
    #              (m * m - n * n) * 2 * m * n * C33
    # Cmat[2, 2] = m * m * n * n * (C11 - C12) - \
    #              m * m * n * n * (C12 - C22) + \
    #              C33 * (m * m - n * n) * (m * m - n * n)

    # # TEST
    # Ctest = np.zeros((3, 3))
    Cmat[0, 0] = C11
    Cmat[0, 1] = C12
    Cmat[1, 0] = C12
    Cmat[1, 1] = C22
    Cmat[2, 2] = C33
