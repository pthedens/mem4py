# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef object sparsityPattern(int [:, ::1] NBar,
                            int [:, ::1] NMem,
                            unsigned int nelemsBar,
                            unsigned int nelemsMem,
                            unsigned int ndof,
                            unsigned int dim)


cdef int addRows(double [:] data,
                 unsigned int [:] indptr,
                 unsigned int ndof,
                 double [:] diagK,
                 double * alpha,
                 double [:] M,
                 double [:] Minv,
                 double [:] sumRowWithDiag,
                 unsigned int iteration,
                 unsigned int dim,
                 str method,
                 double lam) nogil