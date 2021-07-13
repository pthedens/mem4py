# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef int computeNormalVector(int [:, ::1] NMem,
                             double [:] X,
                             double [:] Y,
                             double [:] Z,
                             unsigned int [:] elPressurised,
                             double [:, ::1] normalVector,
                             unsigned int nelemsCable) except -1