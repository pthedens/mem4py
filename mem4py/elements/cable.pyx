# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
cimport cython

# cdef extern from "math.h":
#     double sqrt(double m)
from libc.math cimport sqrt
from libcpp.vector cimport vector


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int cable2DCauchyStress(double [:] X,
                             double [:] Y,
                             int [:, ::1] N,
                             double L0,
                             double ECable,
                             double * strainCable,
                             double * stressCable,
                             Py_ssize_t el,
                             unsigned int nonCompression) nogil:
    """
    compute Cauchy stress (in current configuration) with
    or without allowable compression for 2D cable element
    :param X: 
    :param Y: 
    :param N: 
    :param L0: 
    :param ECable: 
    :param strainCable: 
    :param stressCable: 
    :param el: 
    :param nonCompression: if 1 -> don't allow negative strain
    :return: void
    """

    cdef double X21, Y21, L

    # Nodal distances in xyz system
    X21 = X[N[el, 2]] - X[N[el, 1]]
    Y21 = Y[N[el, 2]] - Y[N[el, 1]]

    # Compute cable length in current configuration
    L = sqrt(X21 * X21 +
             Y21 * Y21)

    strainCable[0] = (L * L - L0 * L0) / (2 * L0 * L0)

    stressCable[0] = ECable * strainCable[0] * L / L0

    if stressCable[0] < 0 and nonCompression == 1:
        stressCable[0] = 0


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int cable2DFintAndK(double [:] X,
                         double [:] Y,
                         int [:, ::1] N,
                         double L0,
                         Py_ssize_t el,
                         double area,
                         double E,
                         unsigned int index,
                         double [:] Fint,
                         double [:] data,
                         double [:] diagK,
                         unsigned int [:] order,
                         double * strainEnergy,
                         unsigned int nonCompression) nogil:
    """
    determine internal force vector and stiffness contribution of
    single 2D cable element with or without allowable compression
    :param X: 
    :param Y: 
    :param N: 
    :param L0: 
    :param el: 
    :param areaCable: 
    :param E: 
    :param index: 
    :param Fint: 
    :param data: 
    :param diagK: 
    :param order: 
    :param strainEnergy: 
    :param nonCompression: if 1 -> don't allow negative strain
    :return: void
    """

    cdef:
        Py_ssize_t dof_1, dof_2, dof_3, dof_4

        double L, X21, Y21, f, S
        double strainCable

    # Find degrees of freedom from current element
    dof_1 = 2 * (N[el, 1] + 1) - 2
    dof_2 = 2 * (N[el, 1] + 1) - 1
    dof_3 = 2 * (N[el, 2] + 1) - 2
    dof_4 = 2 * (N[el, 2] + 1) - 1

    # Nodal distances in xy system
    X21 = X[N[el, 2]] - X[N[el, 1]]
    Y21 = Y[N[el, 2]] - Y[N[el, 1]]

    # Compute cable length in current configuration
    L = sqrt(X21 * X21 +
             Y21 * Y21)

    cableStrainGreen(L, L0, &strainCable)

    S = E * strainCable

    # Compression criterion
    if S < 0 and nonCompression == 1:
        S = 0
    else:
        # Internal force vector
        f = S * area / L0

        Fint[dof_1] -= X21 * f
        Fint[dof_2] -= Y21 * f
        Fint[dof_3] += X21 * f
        Fint[dof_4] += Y21 * f

    # Tangent stiffness matrix
    f = E * area / L0
    data[order[index]]     += f * (X21 * X21 / (L0 * L0) + strainCable)
    data[order[index + 1]] += f * X21 * Y21 / (L0 * L0)
    data[order[index + 2]] -= f * (X21 * X21 / (L0 * L0) + strainCable)
    data[order[index + 3]] -= f * X21 * Y21 / (L0 * L0)

    data[order[index + 4]] += f * Y21 * X21 / (L0 * L0)
    data[order[index + 5]] += f * (Y21 * Y21 / (L0 * L0) + strainCable)
    data[order[index + 6]] -= f * Y21 * X21 / (L0 * L0)
    data[order[index + 7]] -= f * (Y21 * Y21 / (L0 * L0) + strainCable)

    data[order[index + 8]] -= f * (X21 * X21 / (L0 * L0) + strainCable)
    data[order[index + 9]] -= f * X21 * Y21 / (L0 * L0)
    data[order[index + 10]] += f * (X21 * X21 / (L0 * L0) + strainCable)
    data[order[index + 11]] += f * X21 * Y21 / (L0 * L0)

    data[order[index + 12]] -= f * Y21 * X21 / (L0 * L0)
    data[order[index + 13]] -= f * (Y21 * Y21 / (L0 * L0) + strainCable)
    data[order[index + 14]] += f * Y21 * X21 / (L0 * L0)
    data[order[index + 15]] += f * (Y21 * Y21 / (L0 * L0) + strainCable)

    diagK[dof_1] += f * (X21 * X21 / (L0 * L0) + strainCable)
    diagK[dof_2] += f * (Y21 * Y21 / (L0 * L0) + strainCable)
    diagK[dof_3] += f * (X21 * X21 / (L0 * L0) + strainCable)
    diagK[dof_4] += f * (Y21 * Y21 / (L0 * L0) + strainCable)

    strainEnergy[0] = L0 * area * strainCable * strainCable * E


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int cable2D_internal_force_vector(double [:] X,
                                       double [:] Y,
                                       int [:, ::1] N,
                                       double L0,
                                       Py_ssize_t el,
                                       double area,
                                       double E,
                                       double [:] Fint,
                                       unsigned int nonCompression) nogil:

    """
    
    """

    cdef:
        Py_ssize_t dof_1, dof_2, dof_3, dof_4

        double L, X21, Y21, f, S
        double strainCable

    # Find degrees of freedom from current element
    dof_1 = 2 * (N[el, 1] + 1) - 2
    dof_2 = 2 * (N[el, 1] + 1) - 1
    dof_3 = 2 * (N[el, 2] + 1) - 2
    dof_4 = 2 * (N[el, 2] + 1) - 1

    # Nodal distances in xy system
    X21 = X[N[el, 2]] - X[N[el, 1]]
    Y21 = Y[N[el, 2]] - Y[N[el, 1]]

    # Compute cable length in current configuration
    L = sqrt(X21 * X21 +
             Y21 * Y21)

    cableStrainGreen(L, L0, &strainCable)

    S = E * strainCable

    # Compression criterion
    if S < 0 and nonCompression == 1:
        S = 0
    else:
        # Internal force vector
        f = S * area / L0

        Fint[dof_1] -= X21 * f
        Fint[dof_2] -= Y21 * f
        Fint[dof_3] += X21 * f
        Fint[dof_4] += Y21 * f


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int cable3DCauchyStress(double [:] X,
                             double [:] Y,
                             double [:] Z,
                             int [:, ::1] N,
                             double L0,
                             double ECable,
                             double * strainCable,
                             double * stressCable,
                             Py_ssize_t el,
                             unsigned int nonCompression) nogil:
    """
    compute Cauchy stress (in current configuration) with
    or without allowable compression for 3D cable element
    :param X: 
    :param Y: 
    :param Z: 
    :param N: 
    :param L0: 
    :param ECable: 
    :param strainCable: 
    :param stressCable: 
    :param el: 
    :param nonCompression: if 1 -> don't allow negative strain
    :return: void
    """
    cdef:
        double X21, Y21, Z21, L

    # Nodal distances in xyz system
    X21 = X[N[el, 2]] - X[N[el, 1]]
    Y21 = Y[N[el, 2]] - Y[N[el, 1]]
    Z21 = Z[N[el, 2]] - Z[N[el, 1]]

    # Compute cable length in current configuration
    L = sqrt(X21 * X21 +
             Y21 * Y21 +
             Z21 * Z21)

    strainCable[0] = (L * L - L0 * L0) / (2 * L0 * L0)

    if strainCable[0] < 0 and nonCompression == 1:
        stressCable[0] = 0
    else:
        stressCable[0] = ECable * strainCable[0] * L / L0


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int cable3DFintAndK(double [:] X,
                         double [:] Y,
                         double [:] Z,
                         int [:, ::1] N,
                         double L0,
                         Py_ssize_t el,
                         double areaCable,
                         double E,
                         unsigned int index,
                         double [:] Fint,
                         double [:] data,
                         double [:] diagK,
                         unsigned int [:] order,
                         double * SE,
                         unsigned int nonCompression) nogil:
    """
    determine internal force vector and stiffness contribution of
    single 3D cable element with or without allowable compression
    :param X: 
    :param Y: 
    :param Z: 
    :param N: 
    :param L0: 
    :param el: 
    :param areaCable: 
    :param E: 
    :param index: 
    :param Fint: 
    :param data: 
    :param diagK: 
    :param order: 
    :param SE: 
    :param nonCompression: if 1 -> don't allow negative strain
    :return: void
    """

    cdef:
        double L, X21, Y21, Z21, f
        double strainCable
        Py_ssize_t dof_1, dof_2, dof_3, dof_4, dof_5, dof_6

    # Find degrees of freedom from current element
    dof_1 = 3 * (N[el, 1] + 1) - 3
    dof_2 = 3 * (N[el, 1] + 1) - 2
    dof_3 = 3 * (N[el, 1] + 1) - 1
    dof_4 = 3 * (N[el, 2] + 1) - 3
    dof_5 = 3 * (N[el, 2] + 1) - 2
    dof_6 = 3 * (N[el, 2] + 1) - 1

    # Nodal distances in xyz system
    X21 = X[N[el, 2]] - X[N[el, 1]]
    Y21 = Y[N[el, 2]] - Y[N[el, 1]]
    Z21 = Z[N[el, 2]] - Z[N[el, 1]]

    # Compute cable length in current configuration
    L = sqrt(X21 * X21 +
             Y21 * Y21 +
             Z21 * Z21)

    cableStrainGreen(L, L0, &strainCable)

    # Compression criterion
    if strainCable < 0 and nonCompression == 1:
        strainCable = 0
    else:
        # Internal force vector
        f = E * strainCable * areaCable / L0

        Fint[dof_1] -= X21 * f
        Fint[dof_2] -= Y21 * f
        Fint[dof_3] -= Z21 * f
        Fint[dof_4] += X21 * f
        Fint[dof_5] += Y21 * f
        Fint[dof_6] += Z21 * f

    # Tangent stiffness matrix
    f = E * areaCable / L0
    data[order[index]]     += f * (X21 * X21 / (L0 * L0) + strainCable)
    data[order[index + 1]] += f * X21 * Y21 / (L0 * L0)
    data[order[index + 2]] += f * X21 * Z21 / (L0 * L0)
    data[order[index + 3]] -= f * (X21 * X21 / (L0 * L0) + strainCable)
    data[order[index + 4]] -= f * X21 * Y21 / (L0 * L0)
    data[order[index + 5]] -= f * X21 * Z21 / (L0 * L0)

    data[order[index + 6]] += f * Y21 * X21 / (L0 * L0)
    data[order[index + 7]] += f * (Y21 * Y21 / (L0 * L0) + strainCable)
    data[order[index + 8]] += f * Y21 * Z21 / (L0 * L0)
    data[order[index + 9]] -= f * Y21 * X21 / (L0 * L0)
    data[order[index + 10]] -= f * (Y21 * Y21 / (L0 * L0) + strainCable)
    data[order[index + 11]] -= f * Y21 * Z21 / (L0 * L0)

    data[order[index + 12]] += f * Z21 * X21 / (L0 * L0)
    data[order[index + 13]] += f * Z21 * Y21 / (L0 * L0)
    data[order[index + 14]] += f * (Z21 * Z21 / (L0 * L0) + strainCable)
    data[order[index + 15]] -= f * Z21 * X21 / (L0 * L0)
    data[order[index + 16]] -= f * Z21 * Y21 / (L0 * L0)
    data[order[index + 17]] -= f * (Z21 * Z21 / (L0 * L0) + strainCable)

    data[order[index + 18]] -= f * (X21 * X21 / (L0 * L0) + strainCable)
    data[order[index + 19]] -= f * X21 * Y21 / (L0 * L0)
    data[order[index + 20]] -= f * X21 * Z21 / (L0 * L0)
    data[order[index + 21]] += f * (X21 * X21 / (L0 * L0) + strainCable)
    data[order[index + 22]] += f * X21 * Y21 / (L0 * L0)
    data[order[index + 23]] += f * X21 * Z21 / (L0 * L0)

    data[order[index + 24]] -= f * Y21 * X21 / (L0 * L0)
    data[order[index + 25]] -= f * (Y21 * Y21 / (L0 * L0) + strainCable)
    data[order[index + 26]] -= f * Y21 * Z21 / (L0 * L0)
    data[order[index + 27]] += f * Y21 * X21 / (L0 * L0)
    data[order[index + 28]] += f * (Y21 * Y21 / (L0 * L0) + strainCable)
    data[order[index + 29]] += f * Y21 * Z21 / (L0 * L0)

    data[order[index + 30]] -= f * Z21 * X21 / (L0 * L0)
    data[order[index + 31]] -= f * Z21 * Y21 / (L0 * L0)
    data[order[index + 32]] -= f * (Z21 * Z21 / (L0 * L0) + strainCable)
    data[order[index + 33]] += f * Z21 * X21 / (L0 * L0)
    data[order[index + 34]] += f * Z21 * Y21 / (L0 * L0)
    data[order[index + 35]] += f * (Z21 * Z21 / (L0 * L0) + strainCable)

    # diagonal entries of K
    diagK[dof_1] += f * (X21 * X21 / (L0 * L0) + strainCable)
    diagK[dof_2] += f * (Y21 * Y21 / (L0 * L0) + strainCable)
    diagK[dof_3] += f * (Z21 * Z21 / (L0 * L0) + strainCable)
    diagK[dof_4] += f * (X21 * X21 / (L0 * L0) + strainCable)
    diagK[dof_5] += f * (Y21 * Y21 / (L0 * L0) + strainCable)
    diagK[dof_6] += f * (Z21 * Z21 / (L0 * L0) + strainCable)

    SE[0] = areaCable * L0 * strainCable * strainCable * E


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int cable3D_internal_force_vector(double [:] X,
                                       double [:] Y,
                                       double [:] Z,
                                       int [:, ::1] N,
                                       double L0,
                                       Py_ssize_t el,
                                       double area,
                                       double E,
                                       double [:] Fint,
                                       unsigned int nonCompression) nogil:

    """
    
    """

    cdef:
        double L, X21, Y21, Z21, f
        double strainCable
        Py_ssize_t dof_1, dof_2, dof_3, dof_4, dof_5, dof_6

    # Find degrees of freedom from current element
    dof_1 = 3 * (N[el, 1] + 1) - 3
    dof_2 = 3 * (N[el, 1] + 1) - 2
    dof_3 = 3 * (N[el, 1] + 1) - 1
    dof_4 = 3 * (N[el, 2] + 1) - 3
    dof_5 = 3 * (N[el, 2] + 1) - 2
    dof_6 = 3 * (N[el, 2] + 1) - 1

    # Nodal distances in xyz system
    X21 = X[N[el, 2]] - X[N[el, 1]]
    Y21 = Y[N[el, 2]] - Y[N[el, 1]]
    Z21 = Z[N[el, 2]] - Z[N[el, 1]]

    # Compute cable length in current configuration
    L = sqrt(X21 * X21 +
             Y21 * Y21 +
             Z21 * Z21)

    cableStrainGreen(L, L0, &strainCable)

    # Compression criterion
    if strainCable < 0 and nonCompression == 1:
        strainCable = 0
    else:
        # Internal force vector
        f = E * strainCable * area / L0

        Fint[dof_1] -= X21 * f
        Fint[dof_2] -= Y21 * f
        Fint[dof_3] -= Z21 * f
        Fint[dof_4] += X21 * f
        Fint[dof_5] += Y21 * f
        Fint[dof_6] += Z21 * f


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int cableStrainGreen(double L, double L0, double * strainCable) nogil:
    """
    compute Green's strain for single element
    :param L: 
    :param L0: 
    :param strainCable: 
    :return: exception
    """

    strainCable[0] = (L * L - L0 * L0) / (2 * L0 * L0)
