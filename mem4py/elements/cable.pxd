# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef int cable2DCauchyStress(double [:] X,
                             double [:] Y,
                             int [:, ::1] N,
                             double L0,
                             double ECable,
                             double * strainCable,
                             double * stressCable,
                             Py_ssize_t el,
                             unsigned int nonCompression) nogil


cdef int cable2DFintAndK(double [:] X,
                         double [:] Y,
                         int [:, ::1] N,
                         double L0,
                         Py_ssize_t el,
                         double area,
                         double E,
                         unsigned int index,
                         double [:] Fint,
                         double [:] data,
                         double [:] diagK,
                         unsigned int [:] order,
                         double * strainEnergy,
                         unsigned int nonCompression) nogil


cdef int cable3DCauchyStress(double [:] X,
                             double [:] Y,
                             double [:] Z,
                             int [:, ::1] N,
                             double L0,
                             double ECable,
                             double * strainCable,
                             double * stressCable,
                             Py_ssize_t el,
                             unsigned int nonCompression) nogil


cdef int cable3DFintAndK(double [:] X,
                         double [:] Y,
                         double [:] Z,
                         int [:, ::1] N,
                         double L0,
                         Py_ssize_t el,
                         double areaCable,
                         double E,
                         unsigned int index,
                         double [:] Fint,
                         double [:] data,
                         double [:] diagK,
                         unsigned int [:] order,
                         double * SE,
                         unsigned int nonCompression) nogil


cdef int cableStrainGreen(double L, double L0, double * strainCable) nogil


cdef int cable2D_internal_force_vector(double [:] X,
                                       double [:] Y,
                                       int [:, ::1] N,
                                       double L0,
                                       Py_ssize_t el,
                                       double area,
                                       double E,
                                       double [:] Fint,
                                       unsigned int nonCompression) nogil


cdef int cable3D_internal_force_vector(double [:] X,
                                       double [:] Y,
                                       double [:] Z,
                                       int [:, ::1] N,
                                       double L0,
                                       Py_ssize_t el,
                                       double area,
                                       double E,
                                       double [:] Fint,
                                       unsigned int nonCompression) nogil