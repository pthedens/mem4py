# cython: language_level=3
# cython: boundcheck=False
cdef int membrane2DStrain(double [:] X,
                          double [:] Y,
                          int [:, ::1] N,
                          Py_ssize_t el,
                          long double J11,
                          long double J22,
                          long double J12,
                          double [:] ELocal) nogil


cdef int membrane2DKmat(double [:] X,
                        double [:] Y,
                        int [:, ::1] N,
                        double [:] SLocal,
                        double [:] Fint,
                        double t,
                        double area,
                        double p,
                        Py_ssize_t el,
                        long double J11,
                        long double J22,
                        long double J12,
                        double E,
                        double nu,
                        double [:] data,
                        double [:] diagK,
                        unsigned int ind,
                        unsigned int [:] order) nogil


cdef int membrane3DStrain(double [:] X,
                          double [:] Y,
                          double [:] Z,
                          int [:, ::1] N,
                          Py_ssize_t el,
                          long double J11,
                          long double J22,
                          long double J12,
                          double [:] ELocal) nogil


cdef int membrane3DKmat(double [:] X,
                        double [:] Y,
                        double [:] Z,
                        int [:, ::1] N,
                        double [:] SLocal,
                        double [:] Fint,
                        double t,
                        double area,
                        double p,
                        Py_ssize_t el,
                        long double J11,
                        long double J22,
                        long double J12,
                        double E,
                        double nu,
                        double [:] data,
                        double [:] diagK,
                        unsigned int ind,
                        unsigned int [:] order) nogil


# cdef int membrane3DKmatVisc(double [:] X,
#                             double [:] Y,
#                             double [:] Z,
#                             int [:, ::1] N,
#                             double [:] SLocal,
#                             double [:] Fint,
#                             double t,
#                             double area,
#                             double p,
#                             Py_ssize_t el,
#                             long double J11,
#                             long double J22,
#                             long double J12,
#                             double E,
#                             double nu,
#                             double [:] data,
#                             double [:] diagK,
#                             unsigned int ind,
#                             unsigned int [:] order,
#                             double [:] V,
#                             double beta) nogil


cdef int membraneStress(double [:] strainVoigt,
                        double [:] stressVoigt,
                        double * S1,
                        double * S2,
                        double * theta,
                        double E,
                        double nu) nogil


cdef int membraneWrinklingJarasjarungkiat(double [:] ELocal,
                                          double [:] SLocal,
                                          unsigned int [:] state,
                                          double S2,
                                          double theta,
                                          Py_ssize_t el,
                                          double E,
                                          double nu,
                                          double [:] P,
                                          double [:] alpha_array,
                                          unsigned int wrinkling_iter,
                                          unsigned int iter_goal,
                                          double sigma_max) nogil


cdef int membraneWrinklingState(double [:] ELocal,
                                double [:] SLocal,
                                unsigned int [:] state,
                                unsigned int nelemsCable,
                                Py_ssize_t el) nogil


cdef int membrane2D_internal_force_vector(double [:] Fint,
                                          long double J11,
                                          long double J22,
                                          long double J12,
                                          int [:, ::1] N,
                                          Py_ssize_t el,
                                          double [:] X,
                                          double [:] Y,
                                          double [:] SLocal,
                                          double area,
                                          double t) nogil


cdef int membrane3D_internal_force_vector(double [:] Fint,
                                          long double J11,
                                          long double J22,
                                          long double J12,
                                          int [:, ::1] N,
                                          Py_ssize_t el,
                                          double [:] X,
                                          double [:] Y,
                                          double [:] Z,
                                          double [:] SLocal,
                                          double area,
                                          double t,
                                          double E,
                                          double nu,
                                          double [:] V,
                                          double beta) nogil