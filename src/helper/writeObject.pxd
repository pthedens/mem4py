cdef void writeVector2D(double [:], unsigned int, str, object)


cdef void writeVectorNode3D(double [:], unsigned int, str, object)


cdef void writeVectorNode2D(double [:] u, unsigned int nnodes, str name, object fout)


cdef void writeVectorElement(double [:, ::1], str, unsigned int , object)


cdef void writeScalarElement(double [:], str, unsigned int , object)


cdef void writeScalarElementInt(unsigned int [:], str, unsigned int , object)