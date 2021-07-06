# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef void area(double [:],
               double [:],
               double [:],
               int [:, ::1],
               unsigned int,
               double [:])


cdef void areaSingle(double [:],
                     double [:],
                     double [:],
                     int [:],
                     double *)