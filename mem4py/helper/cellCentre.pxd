# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef int cellCentre3D(unsigned int [:] dofList,
                      double [:] X,
                      double [:] Y,
                      double [:] Z,
                      int [:, ::1] Nm,
                      double [:, ::1] cellCentre) except -1


cdef int cellCentre2D(unsigned int nelems,
                      double [:] X,
                      double [:] Y,
                      int [:, ::1] Nb,
                      double [:, ::1] cellCentre) except -1