# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef initialiseDirichletBC(unsigned int [:],
                           unsigned int)

cdef void initialiseLoadBC(int [:, ::1],
                           unsigned int [:],
                           unsigned int,
                           unsigned int [:],
                           unsigned int [:])

cdef void correctBC(double [:],
                      int [:])

cdef object dirichlet_zero_matrix_modification(object,
                                               int [:])