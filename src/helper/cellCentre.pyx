# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cimport cython


cdef int cellCentre3D(unsigned int [:] dofList,
                      double [:] X,
                      double [:] Y,
                      double [:] Z,
                      int [:, ::1] Nm,
                      double [:, ::1] cellCentre) except -1:

    cdef unsigned int el, ind = 0

    for el in dofList:

        cellCentre[ind, 0] = (X[Nm[el, 1]] + X[Nm[el, 2]] + X[Nm[el, 3]]) / 3
        cellCentre[ind, 1] = (Y[Nm[el, 1]] + Y[Nm[el, 2]] + Y[Nm[el, 3]]) / 3
        cellCentre[ind, 2] = (Z[Nm[el, 1]] + Z[Nm[el, 2]] + Z[Nm[el, 3]]) / 3
        ind += 1


cdef int cellCentre2D(unsigned int nelems,
                      double [:] X,
                      double [:] Y,
                      int [:, ::1] Nb,
                      double [:, ::1] cellCentre) except -1:

    cdef unsigned int el, ind = 0

    for el in range(nelems):

        cellCentre[ind, 0] = (X[Nb[el, 1]] + X[Nb[el, 2]]) / 2
        cellCentre[ind, 1] = (Y[Nb[el, 1]] + Y[Nb[el, 2]]) / 2
        ind += 1