# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cimport cython

cdef extern from "math.h":
    double sqrt(double m)

cdef int computeNormalVector(int [:, ::1] NMem,
                             double [:] X,
                             double [:] Y,
                             double [:] Z,
                             unsigned int [:] elPressurised,
                             double [:, ::1] normalVector,
                             unsigned int nelemsCable) except -1:
    """
    :param NMem: 
    :param X: 
    :param Y: 
    :param Z: 
    :param normalVector: 
    :return: void 
    """
    cdef:
        double Nnorm
        Py_ssize_t el

    for el in range(len(NMem)):

        Nnorm = sqrt(((Y[NMem[el, 2]] - Y[NMem[el, 1]]) * (Z[NMem[el, 3]] - Z[NMem[el, 1]]) -
                      (Z[NMem[el, 2]] - Z[NMem[el, 1]]) * (Y[NMem[el, 3]] - Y[NMem[el, 1]])) *
                     ((Y[NMem[el, 2]] - Y[NMem[el, 1]]) * (Z[NMem[el, 3]] - Z[NMem[el, 1]]) -
                      (Z[NMem[el, 2]] - Z[NMem[el, 1]]) * (Y[NMem[el, 3]] - Y[NMem[el, 1]])) +

                     ((Z[NMem[el, 2]] - Z[NMem[el, 1]]) * (X[NMem[el, 3]] - X[NMem[el, 1]]) -
                      (X[NMem[el, 2]] - X[NMem[el, 1]]) * (Z[NMem[el, 3]] - Z[NMem[el, 1]])) *
                     ((Z[NMem[el, 2]] - Z[NMem[el, 1]]) * (X[NMem[el, 3]] - X[NMem[el, 1]]) -
                      (X[NMem[el, 2]] - X[NMem[el, 1]]) * (Z[NMem[el, 3]] - Z[NMem[el, 1]])) +

                     ((X[NMem[el, 2]] - X[NMem[el, 1]]) * (Y[NMem[el, 3]] - Y[NMem[el, 1]]) -
                      (Y[NMem[el, 2]] - Y[NMem[el, 1]]) * (X[NMem[el, 3]] - X[NMem[el, 1]])) *
                     ((X[NMem[el, 2]] - X[NMem[el, 1]]) * (Y[NMem[el, 3]] - Y[NMem[el, 1]]) -
                      (Y[NMem[el, 2]] - Y[NMem[el, 1]]) * (X[NMem[el, 3]] - X[NMem[el, 1]])))

        normalVector[nelemsCable + el, 0] = ((Y[NMem[el, 2]] - Y[NMem[el, 1]]) *
                                             (Z[NMem[el, 3]] - Z[NMem[el, 1]]) -
                                             (Z[NMem[el, 2]] - Z[NMem[el, 1]]) *
                                             (Y[NMem[el, 3]] - Y[NMem[el, 1]])) / Nnorm
        normalVector[nelemsCable + el, 1] = ((Z[NMem[el, 2]] - Z[NMem[el, 1]]) *
                                             (X[NMem[el, 3]] - X[NMem[el, 1]]) -
                                             (X[NMem[el, 2]] - X[NMem[el, 1]]) *
                                             (Z[NMem[el, 3]] - Z[NMem[el, 1]])) / Nnorm
        normalVector[nelemsCable + el, 2] = ((X[NMem[el, 2]] - X[NMem[el, 1]]) *
                                             (Y[NMem[el, 3]] - Y[NMem[el, 1]]) -
                                             (Y[NMem[el, 2]] - Y[NMem[el, 1]]) *
                                             (X[NMem[el, 3]] - X[NMem[el, 1]])) / Nnorm