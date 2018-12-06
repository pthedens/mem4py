cimport cython

cdef extern from "math.h":
    double sqrt(double m)

# Compute the area and normal vector of given triangle
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void area(double [:] X,
               double [:] Y,
               double [:] Z,
               int [:, ::1] N,
               unsigned int nelems,
               double [:] areaVec):

    cdef unsigned int el
    cdef double X1, X2, X3, Y1, Y2, Y3

    for el in range(nelems):

        # AB = (X1, X2, X3), AC = (Y1, Y2, Y3)
        X1 = X[N[el, 2]] - X[N[el, 1]]
        X2 = Y[N[el, 2]] - Y[N[el, 1]]
        X3 = Z[N[el, 2]] - Z[N[el, 1]]
        Y1 = X[N[el, 3]] - X[N[el, 1]]
        Y2 = Y[N[el, 3]] - Y[N[el, 1]]
        Y3 = Z[N[el, 3]] - Z[N[el, 1]]

        areaVec[el] = 0.5 * sqrt((X2 * Y3 - X3 * Y2) * (X2 * Y3 - X3 * Y2) +
                                 (X3 * Y1 - X1 * Y3) * (X3 * Y1 - X1 * Y3) +
                                 (X1 * Y2 - X2 * Y1) * (X1 * Y2 - X2 * Y1))


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void areaSingle(double [:] X,
                     double [:] Y,
                     double [:] Z,
                     int [:] N,
                     double * A):

    cdef double X1, X2, X3, Y1, Y2, Y3

    # AB = (X1, X2, X3), AC = (Y1, Y2, Y3)
    X1 = X[N[2]] - X[N[1]]
    X2 = Y[N[2]] - Y[N[1]]
    X3 = Z[N[2]] - Z[N[1]]
    Y1 = X[N[3]] - X[N[1]]
    Y2 = Y[N[3]] - Y[N[1]]
    Y3 = Z[N[3]] - Z[N[1]]

    A[0] = 0.5 * sqrt((X2 * Y3 - X3 * Y2) * (X2 * Y3 - X3 * Y2) +
                             (X3 * Y1 - X1 * Y3) * (X3 * Y1 - X1 * Y3) +
                             (X1 * Y2 - X2 * Y1) * (X1 * Y2 - X2 * Y1))