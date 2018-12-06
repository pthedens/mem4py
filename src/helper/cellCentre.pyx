cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void cellCentre(unsigned int [:] dofList,
                     double [:] X,
                     double [:] Y,
                     double [:] Z,
                     int [:, ::1] Nm,
                     double [:, ::1] cellCentre):

    cdef unsigned int el

    for el in dofList:

        cellCentre[el, 0] = (X[Nm[el, 1]] + X[Nm[el, 2]] + X[Nm[el, 3]]) / 3
        cellCentre[el, 1] = (Y[Nm[el, 1]] + Y[Nm[el, 2]] + Y[Nm[el, 3]]) / 3
        cellCentre[el, 2] = (Z[Nm[el, 1]] + Z[Nm[el, 2]] + Z[Nm[el, 3]]) / 3