cdef initialiseDirichletBC(unsigned int [:],
                           unsigned int)

cdef void initialiseLoadBC(int [:, ::1],
                           unsigned int [:],
                           unsigned int,
                           unsigned int [:],
                           unsigned int [:])

cdef void correctBC3D(double [:],
                      int [:])

cdef object dirichlet_zero_matrix_modification(object,
                                               int [:])