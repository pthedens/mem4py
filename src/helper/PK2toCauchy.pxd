# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef int PK2toCauchy(double [:] X,
                     double [:] Y,
                     double [:] Z,
                     double J11,
                     double J22,
                     double J12,
                     int [:, ::1] N,
                     unsigned int el,
                     double A0,
                     double [:, :] PK2,
                     double [:] cauchyLocal) except -1