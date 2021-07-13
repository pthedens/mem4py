# cython: language_level=3
# cython: boundcheck=False
import numpy as np
cimport numpy as np
cimport cython

# cdef extern from "math.h":
#     double sqrt(double m)
#     double cos(double m)
#     double sin(double m)
#     double atan2(double m, double n)
#     double fabs(double m)
from libc.math cimport sqrt, sin, cos, atan2, fabs


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int membrane2DStrain(double [:] X,
                          double [:] Y,
                          int [:, ::1] N,
                          Py_ssize_t el,
                          long double J11,
                          long double J22,
                          long double J12,
                          double [:] ELocal) nogil:

    cdef:
        double g11, g12, g22

        double X21, X31, Y21, Y31

    with cython.boundscheck(False):
        X21 = X[N[el, 2]] - X[N[el, 1]]
        X31 = X[N[el, 3]] - X[N[el, 1]]

        Y21 = Y[N[el, 2]] - Y[N[el, 1]]
        Y31 = Y[N[el, 3]] - Y[N[el, 1]]

        # covariant components of the metric tensor in current configuration g_ab
        g11 = X21 * X21 + Y21 * Y21
        g12 = X21 * X31 + Y21 * Y31
        g22 = X31 * X31 + Y31 * Y31

        # local strain (Cartesian coordinates), ELocal = Q * ECurv, ECurv = 0.5 * (g_ab - G_ab)
        ELocal[0] = 0.5 * g11 / (J11 * J11) - 0.5
        ELocal[1] = 0.5 * (g22 * J11 -
                           2 * g12 * J12 +
                           g11 * J12 * J12 / J11) / (J11 * J22 * J22) - \
                    0.5
        ELocal[2] = (g12 - g11 * J12 / J11) / (J11 * J22) # gamma_xy


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
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
                        unsigned int [:] order) nogil:

    cdef:
        long double Q11 = 1 / (J11 * J11)
        long double Q21 = (J12 * J12) / (J11 * J11 * J22 * J22)
        long double Q31 = - 2 * J12 / (J11 * J11 * J22)
        long double Q22 = 1 / (J22 * J22)
        long double Q23 = - J12 / (J11 * J22 * J22)
        long double Q33 = 1 / (J11 * J22)

        double s0, s1, s2, X21, X31, Y21, Y31

        # local constitutive matrix parameters
        double C11 = E / (1 - nu * nu)
        double C12 = nu * E / (1 - nu * nu)
        double C22 = E / (1 - nu * nu)
        double C33 = 0.5 * (1 - nu) * E / (1 - nu * nu)

        Py_ssize_t dof_1, dof_2, dof_3, dof_4, dof_5, dof_6

    with cython.boundscheck(False):

        s0 = Q11 * SLocal[0] + Q21 * SLocal[1] + Q31 * SLocal[2]
        s1 = Q22 * SLocal[1]
        s2 = Q23 * SLocal[1] + Q33 * SLocal[2]

        X21 = X[N[el, 2]] - X[N[el, 1]]
        X31 = X[N[el, 3]] - X[N[el, 1]]

        Y21 = Y[N[el, 2]] - Y[N[el, 1]]
        Y31 = Y[N[el, 3]] - Y[N[el, 1]]

        # Find degrees of freedom from current element
        dof_1 = 2 * (N[el, 1] + 1) - 2
        dof_2 = 2 * (N[el, 1] + 1) - 1
        dof_3 = 2 * (N[el, 2] + 1) - 2
        dof_4 = 2 * (N[el, 2] + 1) - 1
        dof_5 = 2 * (N[el, 3] + 1) - 2
        dof_6 = 2 * (N[el, 3] + 1) - 1

        data[order[ind]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))* (-Q31*X21 + Q33*(-X21 - X31)) -
                                    Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) +
                                    s0 + s1 + 2*s2 + (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))*
                                    (-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))
        data[order[ind + 1]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(-Q31*Y21 + Q33*(-Y21 - Y31)) -
                                        Q11*Y21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) +
                                        (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))*
                                        (-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))
        data[order[ind + 2]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*X21 + Q33*X31) +
                                        Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) -
                                        s0 - s2 + (Q21*X21 + Q23*X31)*
                                        (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 3]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*Y21 + Q33*Y31) +
                                        Q11*Y21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) +
                                        (Q21*Y21 + Q23*Y31)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 4]] += area*t*(C33*Q33*X21*(-Q31*X21 + Q33*(-X21 - X31)) - s1 - s2 + (Q22*X31 + Q23*X21)*
                                        (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 5]] += area*t*(C33*Q33*Y21*(-Q31*X21 + Q33*(-X21 - X31)) + (Q22*Y31 + Q23*Y21)*
                                        (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))

        data[order[ind + 6]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(-Q31*Y21 + Q33*(-Y21 - Y31)) -
                                        Q11*X21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) +
                                        (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))*
                                        (-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))
        data[order[ind + 7]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(-Q31*Y21 + Q33*(-Y21 - Y31)) -
                                        Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) +
                                        s0 + s1 + 2*s2 + (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))*
                                        (-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))
        data[order[ind + 8]] += area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Y21 + Q33*(-Y21 - Y31)) +
                                        Q11*X21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) +
                                        (Q21*X21 + Q23*X31)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 9]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(Q31*Y21 + Q33*Y31) +
                                        Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) -
                                        s0 - s2 + (Q21*Y21 + Q23*Y31)*
                                        (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 10]] += area*t*(C33*Q33*X21*(-Q31*Y21 + Q33*(-Y21 - Y31)) + (Q22*X31 + Q23*X21)*
                                         (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 11]] += area*t*(C33*Q33*Y21*(-Q31*Y21 + Q33*(-Y21 - Y31)) - s1 - s2 +
                                         (Q22*Y31 + Q23*Y21)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))

        data[order[ind + 12]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*X21 + Q33*X31) -
                                         Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) - s0 - s2 +
                                         (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))
        data[order[ind + 13]] += area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Y21 + Q33*(-Y21 - Y31)) -
                                         Q11*Y21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
                                         (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))
        data[order[ind + 14]] += area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*X21 + Q33*X31) +
                                         Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
                                         s0 + (Q21*X21 + Q23*X31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 15]] += area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Y21 + Q33*Y31) + Q11*Y21*
                                         (C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) + (Q21*Y21 + Q23*Y31)*
                                         (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 16]] += area*t*(C33*Q33*X21*(Q31*X21 + Q33*X31) + s2 + (Q22*X31 + Q23*X21)*
                                         (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 17]] += area*t*(C33*Q33*Y21*(Q31*X21 + Q33*X31) + (Q22*Y31 + Q23*Y21)*
                                         (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))

        data[order[ind + 18]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*Y21 + Q33*Y31) - Q11*X21*
                                         (C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))*
                                         (-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))
        data[order[ind + 19]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(Q31*Y21 + Q33*Y31) -
                                         Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) - s0 - s2 +
                                         (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))
        data[order[ind + 20]] += area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Y21 + Q33*Y31) + Q11*X21*
                                         (C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + (Q21*X21 + Q23*X31)*
                                         (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 21]] += area*t*(C33*(Q31*Y21 + Q33*Y31)*(Q31*Y21 + Q33*Y31) +
                                         Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
                                         s0 + (Q21*Y21 + Q23*Y31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 22]] += area*t*(C33*Q33*X21*(Q31*Y21 + Q33*Y31) + (Q22*X31 + Q23*X21)*
                                         (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 23]] += area*t*(C33*Q33*Y21*(Q31*Y21 + Q33*Y31) + s2 + (Q22*Y31 + Q23*Y21)*
                                         (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))

        data[order[ind + 24]] += area*t*(-C12*Q11*X21*(Q22*X31 + Q23*X21) + C22*(Q22*X31 + Q23*X21)*
                                         (-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + C33*Q33*X21*
                                         (-Q31*X21 + Q33*(-X21 - X31)) - s1 - s2)
        data[order[ind + 25]] += area*t*(-C12*Q11*Y21*(Q22*X31 + Q23*X21) + C22*(Q22*X31 + Q23*X21)*
                                         (-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) +
                                         C33*Q33*X21*(-Q31*Y21 + Q33*(-Y21 - Y31)))
        data[order[ind + 26]] += area*t*(C12*Q11*X21*(Q22*X31 + Q23*X21) + C22*(Q21*X21 + Q23*X31)*(Q22*X31 + Q23*X21) +
                                         C33*Q33*X21*(Q31*X21 + Q33*X31) + s2)
        data[order[ind + 27]] += area*t*(C12*Q11*Y21*(Q22*X31 + Q23*X21) + C22*(Q21*Y21 + Q23*Y31)*(Q22*X31 + Q23*X21) +
                                         C33*Q33*X21*(Q31*Y21 + Q33*Y31))
        data[order[ind + 28]] += area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*X31 + Q23*X21) + C33*Q33*Q33*X21*X21 + s1)
        data[order[ind + 29]] += area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Y31 + Q23*Y21) + C33*Q33*Q33*X21*Y21)

        data[order[ind + 30]] += area*t*(-C12*Q11*X21*(Q22*Y31 + Q23*Y21) + C22*(Q22*Y31 + Q23*Y21)*
                                         (-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) +
                                         C33*Q33*Y21*(-Q31*X21 + Q33*(-X21 - X31)))
        data[order[ind + 31]] += area*t*(-C12*Q11*Y21*(Q22*Y31 + Q23*Y21) + C22*(Q22*Y31 + Q23*Y21)*
                                         (-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) +
                                         C33*Q33*Y21*(-Q31*Y21 + Q33*(-Y21 - Y31)) - s1 - s2)
        data[order[ind + 32]] += area*t*(C12*Q11*X21*(Q22*Y31 + Q23*Y21) + C22*(Q21*X21 + Q23*X31)*(Q22*Y31 + Q23*Y21) +
                                         C33*Q33*Y21*(Q31*X21 + Q33*X31))
        data[order[ind + 33]] += area*t*(C12*Q11*Y21*(Q22*Y31 + Q23*Y21) + C22*(Q21*Y21 + Q23*Y31)*(Q22*Y31 + Q23*Y21) +
                                         C33*Q33*Y21*(Q31*Y21 + Q33*Y31) + s2)
        data[order[ind + 34]] += area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Y31 + Q23*Y21) + C33*Q33*Q33*X21*Y21)
        data[order[ind + 35]] += area*t*(C22*(Q22*Y31 + Q23*Y21)*(Q22*Y31 + Q23*Y21) + C33*Q33*Q33*Y21*Y21 + s1)

        Fint[dof_1] += area*t*(-Q11*SLocal[0]*X21 + SLocal[1]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) +
                              SLocal[2]*(-Q31*X21 + Q33*(-X21 - X31)))
        Fint[dof_2] += area*t*(-Q11*SLocal[0]*Y21 + SLocal[1]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) +
                              SLocal[2]*(-Q31*Y21 + Q33*(-Y21 - Y31)))
        Fint[dof_3] += area*t*(Q11*SLocal[0]*X21 + SLocal[1]*(Q21*X21 + Q23*X31) + SLocal[2]*(Q31*X21 + Q33*X31))
        Fint[dof_4] += area*t*(Q11*SLocal[0]*Y21 + SLocal[1]*(Q21*Y21 + Q23*Y31) + SLocal[2]*(Q31*Y21 + Q33*Y31))
        Fint[dof_5] += area*t*(Q33*SLocal[2]*X21 + SLocal[1]*(Q22*X31 + Q23*X21))
        Fint[dof_6] += area*t*(Q33*SLocal[2]*Y21 + SLocal[1]*(Q22*Y31 + Q23*Y21))

        diagK[dof_1] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(-Q31*X21 + Q33*(-X21 - X31)) -
                                       Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) +
                                       s0 + s1 + 2*s2 + (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))*
                                       (-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))
        diagK[dof_2] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(-Q31*Y21 + Q33*(-Y21 - Y31)) -
                                       Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) +
                                       s0 + s1 + 2*s2 + (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))*
                                       (-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))
        diagK[dof_3] += area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*X21 + Q33*X31) +
                                       Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
                                       s0 + (Q21*X21 + Q23*X31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        diagK[dof_4] += area*t*(C33*(Q31*Y21 + Q33*Y31)*(Q31*Y21 + Q33*Y31) +
                                         Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
                                         s0 + (Q21*Y21 + Q23*Y31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        diagK[dof_5] += area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*X31 + Q23*X21) + C33*Q33*Q33*X21*X21 + s1)
        diagK[dof_6] += area*t*(C22*(Q22*Y31 + Q23*Y21)*(Q22*Y31 + Q23*Y21) + C33*Q33*Q33*Y21*Y21 + s1)


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
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
                                          double t) nogil:

    cdef:
        long double Q11 = 1 / (J11 * J11)
        long double Q21 = (J12 * J12) / (J11 * J11 * J22 * J22)
        long double Q31 = - 2 * J12 / (J11 * J11 * J22)
        long double Q22 = 1 / (J22 * J22)
        long double Q23 = - J12 / (J11 * J22 * J22)
        long double Q33 = 1 / (J11 * J22)

        double X21, X31, Y21, Y31

        Py_ssize_t dof_1, dof_2, dof_3, dof_4, dof_5, dof_6

    with cython.boundscheck(False):

        X21 = X[N[el, 2]] - X[N[el, 1]]
        X31 = X[N[el, 3]] - X[N[el, 1]]

        Y21 = Y[N[el, 2]] - Y[N[el, 1]]
        Y31 = Y[N[el, 3]] - Y[N[el, 1]]

        # Find degrees of freedom from current element
        dof_1 = 2 * (N[el, 1] + 1) - 2
        dof_2 = 2 * (N[el, 1] + 1) - 1
        dof_3 = 2 * (N[el, 2] + 1) - 2
        dof_4 = 2 * (N[el, 2] + 1) - 1
        dof_5 = 2 * (N[el, 3] + 1) - 2
        dof_6 = 2 * (N[el, 3] + 1) - 1

        # internal force vector
        Fint[dof_1] += area*t*(-Q11*SLocal[0]*X21 + SLocal[1]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) +
                              SLocal[2]*(-Q31*X21 + Q33*(-X21 - X31)))
        Fint[dof_2] += area*t*(-Q11*SLocal[0]*Y21 + SLocal[1]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) +
                              SLocal[2]*(-Q31*Y21 + Q33*(-Y21 - Y31)))
        Fint[dof_3] += area*t*(Q11*SLocal[0]*X21 + SLocal[1]*(Q21*X21 + Q23*X31) + SLocal[2]*(Q31*X21 + Q33*X31))
        Fint[dof_4] += area*t*(Q11*SLocal[0]*Y21 + SLocal[1]*(Q21*Y21 + Q23*Y31) + SLocal[2]*(Q31*Y21 + Q33*Y31))
        Fint[dof_5] += area*t*(Q33*SLocal[2]*X21 + SLocal[1]*(Q22*X31 + Q23*X21))
        Fint[dof_6] += area*t*(Q33*SLocal[2]*Y21 + SLocal[1]*(Q22*Y31 + Q23*Y21))


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int membrane3DStrain(double [:] X,
                          double [:] Y,
                          double [:] Z,
                          int [:, ::1] N,
                          Py_ssize_t el,
                          long double J11,
                          long double J22,
                          long double J12,
                          double [:] ELocal) nogil:

    cdef:
        long double g11, g12, g22

        double X21, X31, Y21, Y31, Z21, Z31

    with cython.boundscheck(False):

        X21 = X[N[el, 2]] - X[N[el, 1]]
        X31 = X[N[el, 3]] - X[N[el, 1]]

        Y21 = Y[N[el, 2]] - Y[N[el, 1]]
        Y31 = Y[N[el, 3]] - Y[N[el, 1]]

        Z21 = Z[N[el, 2]] - Z[N[el, 1]]
        Z31 = Z[N[el, 3]] - Z[N[el, 1]]

        # covariant components of the metric tensor in current configuration
        g11 = X21 * X21 + Y21 * Y21 + Z21 * Z21
        g12 = X21 * X31 + Y21 * Y31 + Z21 * Z31
        g22 = X31 * X31 + Y31 * Y31 + Z31 * Z31

        # local strain (Cartesian coordinates), ELocal = Q * ECurv, ECurv = 0.5 * (g_ab - G_ab)
        ELocal[0] = 0.5 * g11 / (J11 * J11) - 0.5
        ELocal[1] = 0.5 * (g22 * J11 -
                           2 * g12 * J12 +
                           g11 * J12 * J12 / J11) / (J11 * J22 * J22) - \
                    0.5
        ELocal[2] = (g12 - g11 * J12 / J11) / (J11 * J22) # gamma_xy


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
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
                        unsigned int [:] order) nogil:

    cdef:
        
        double Q11 = 1 / (J11 * J11)
        double Q21 = (J12 * J12) / (J11 * J11 * J22 * J22)
        double Q31 = - 2 * J12 / (J11 * J11 * J22)
        double Q22 = 1 / (J22 * J22)
        double Q23 = - J12 / (J11 * J22 * J22)
        double Q33 = 1 / (J11 * J22)

        double s_0, s_1, s_2, s1, s2, s3

        double X21, X31, X23, Y21, Y31, Y23, Z21, Z31, Z23

        # local constitutive matrix parameters
        double C11 = E / (1 - nu * nu)
        double C12 = nu * E / (1 - nu * nu)
        double C22 = E / (1 - nu * nu)
        double C33 = 0.5 * (1 - nu) * E / (1 - nu * nu)

        Py_ssize_t dof_1, dof_2, dof_3, dof_4, dof_5, dof_6, dof_7, dof_8, dof_9
        
    with cython.boundscheck(False):

        s_0 = Q11 * SLocal[0] + Q21 * SLocal[1] + Q31 * SLocal[2]
        s_1 = Q22 * SLocal[1]
        s_2 = Q23 * SLocal[1] + Q33 * SLocal[2]

        s1 = (s_0 + s_1 + 2 * s_2)
        s2 = - (s_0 + s_2)
        s3 = - (s_1 + s_2)

        X21 = X[N[el, 2]] - X[N[el, 1]]
        X31 = X[N[el, 3]] - X[N[el, 1]]
        X23 = X[N[el, 2]] - X[N[el, 3]]

        Y21 = Y[N[el, 2]] - Y[N[el, 1]]
        Y31 = Y[N[el, 3]] - Y[N[el, 1]]
        Y23 = Y[N[el, 2]] - Y[N[el, 3]]

        Z21 = Z[N[el, 2]] - Z[N[el, 1]]
        Z31 = Z[N[el, 3]] - Z[N[el, 1]]
        Z23 = Z[N[el, 2]] - Z[N[el, 3]]

        # Find degrees of freedom from current element
        dof_1 = 3 * (N[el, 1] + 1) - 3
        dof_2 = 3 * (N[el, 1] + 1) - 2
        dof_3 = 3 * (N[el, 1] + 1) - 1
        dof_4 = 3 * (N[el, 2] + 1) - 3
        dof_5 = 3 * (N[el, 2] + 1) - 2
        dof_6 = 3 * (N[el, 2] + 1) - 1
        dof_7 = 3 * (N[el, 3] + 1) - 3
        dof_8 = 3 * (N[el, 3] + 1) - 2
        dof_9 = 3 * (N[el, 3] + 1) - 1

        Fint[dof_1] += area*t*(-Q11*SLocal[0]*X21 + SLocal[1]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + SLocal[2]*(-Q31*X21 + Q33*(-X21 - X31)))
        Fint[dof_2] += area*t*(-Q11*SLocal[0]*Y21 + SLocal[1]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + SLocal[2]*(-Q31*Y21 + Q33*(-Y21 - Y31)))
        Fint[dof_3] += area*t*(-Q11*SLocal[0]*Z21 + SLocal[1]*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)) + SLocal[2]*(-Q31*Z21 + Q33*(-Z21 - Z31)))
        Fint[dof_4] += area*t*(Q11*SLocal[0]*X21 + SLocal[1]*(Q21*X21 + Q23*X31) + SLocal[2]*(Q31*X21 + Q33*X31))
        Fint[dof_5] += area*t*(Q11*SLocal[0]*Y21 + SLocal[1]*(Q21*Y21 + Q23*Y31) + SLocal[2]*(Q31*Y21 + Q33*Y31))
        Fint[dof_6] += area*t*(Q11*SLocal[0]*Z21 + SLocal[1]*(Q21*Z21 + Q23*Z31) + SLocal[2]*(Q31*Z21 + Q33*Z31))
        Fint[dof_7] += area*t*(Q33*SLocal[2]*X21 + SLocal[1]*(Q22*X31 + Q23*X21))
        Fint[dof_8] += area*t*(Q33*SLocal[2]*Y21 + SLocal[1]*(Q22*Y31 + Q23*Y21))
        Fint[dof_9] += area*t*(Q33*SLocal[2]*Z21 + SLocal[1]*(Q22*Z31 + Q23*Z21))

        data[order[ind + 0]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))**2 - Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + s1 + (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))
        data[order[ind + 1]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(-Q31*Y21 + Q33*(-Y21 - Y31)) - Q11*Y21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + p*(-Z21 + Z31)/6
        data[order[ind + 2]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(-Q31*Z21 + Q33*(-Z21 - Z31)) - Q11*Z21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + p*(Y21 - Y31)/6
        data[order[ind + 3]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*X21 + Q33*X31) + Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + s2 + (Q21*X21 + Q23*X31)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 4]] += -Z31*p/6 + area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*Y21 + Q33*Y31) + Q11*Y21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + (Q21*Y21 + Q23*Y31)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 5]] += Y31*p/6 + area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*Z21 + Q33*Z31) + Q11*Z21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + (Q21*Z21 + Q23*Z31)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 6]] += area*t*(C33*Q33*X21*(-Q31*X21 + Q33*(-X21 - X31)) + s3 + (Q22*X31 + Q23*X21)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 7]] += Z21*p/6 + area*t*(C33*Q33*Y21*(-Q31*X21 + Q33*(-X21 - X31)) + (Q22*Y31 + Q23*Y21)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 8]] += -Y21*p/6 + area*t*(C33*Q33*Z21*(-Q31*X21 + Q33*(-X21 - X31)) + (Q22*Z31 + Q23*Z21)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))))
        data[order[ind + 9]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(-Q31*Y21 + Q33*(-Y21 - Y31)) - Q11*X21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + p*(Z21 - Z31)/6
        data[order[ind + 10]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))**2 - Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + s1 + (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))
        data[order[ind + 11]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(-Q31*Z21 + Q33*(-Z21 - Z31)) - Q11*Z21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + p*(-X21 + X31)/6
        data[order[ind + 12]] += Z31*p/6 + area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Y21 + Q33*(-Y21 - Y31)) + Q11*X21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + (Q21*X21 + Q23*X31)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 13]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(Q31*Y21 + Q33*Y31) + Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + s2 + (Q21*Y21 + Q23*Y31)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 14]] += -X31*p/6 + area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(Q31*Z21 + Q33*Z31) + Q11*Z21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + (Q21*Z21 + Q23*Z31)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 15]] += -Z21*p/6 + area*t*(C33*Q33*X21*(-Q31*Y21 + Q33*(-Y21 - Y31)) + (Q22*X31 + Q23*X21)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 16]] += area*t*(C33*Q33*Y21*(-Q31*Y21 + Q33*(-Y21 - Y31)) + s3 + (Q22*Y31 + Q23*Y21)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 17]] += X21*p/6 + area*t*(C33*Q33*Z21*(-Q31*Y21 + Q33*(-Y21 - Y31)) + (Q22*Z31 + Q23*Z21)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))))
        data[order[ind + 18]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(-Q31*Z21 + Q33*(-Z21 - Z31)) - Q11*X21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + (-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)))*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + p*(-Y21 + Y31)/6
        data[order[ind + 19]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(-Q31*Z21 + Q33*(-Z21 - Z31)) - Q11*Y21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + (-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + p*(X21 - X31)/6
        data[order[ind + 20]] += area*t*(C33*(-Q31*Z21 + Q33*(-Z21 - Z31))**2 - Q11*Z21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + s1 + (-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)))*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)))
        data[order[ind + 21]] += -Y31*p/6 + area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Z21 + Q33*(-Z21 - Z31)) + Q11*X21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + (Q21*X21 + Q23*X31)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))))
        data[order[ind + 22]] += X31*p/6 + area*t*(C33*(Q31*Y21 + Q33*Y31)*(-Q31*Z21 + Q33*(-Z21 - Z31)) + Q11*Y21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + (Q21*Y21 + Q23*Y31)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))))
        data[order[ind + 23]] += area*t*(C33*(-Q31*Z21 + Q33*(-Z21 - Z31))*(Q31*Z21 + Q33*Z31) + Q11*Z21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + s2 + (Q21*Z21 + Q23*Z31)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))))
        data[order[ind + 24]] += Y21*p/6 + area*t*(C33*Q33*X21*(-Q31*Z21 + Q33*(-Z21 - Z31)) + (Q22*X31 + Q23*X21)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))))
        data[order[ind + 25]] += -X21*p/6 + area*t*(C33*Q33*Y21*(-Q31*Z21 + Q33*(-Z21 - Z31)) + (Q22*Y31 + Q23*Y21)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))))
        data[order[ind + 26]] += area*t*(C33*Q33*Z21*(-Q31*Z21 + Q33*(-Z21 - Z31)) + s3 + (Q22*Z31 + Q23*Z21)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))))
        data[order[ind + 27]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*X21 + Q33*X31) - Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) + s2 + (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))
        data[order[ind + 28]] += area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Y21 + Q33*(-Y21 - Y31)) - Q11*Y21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) + (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + p*(-Z21 + Z31)/6
        data[order[ind + 29]] += area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Z21 + Q33*(-Z21 - Z31)) - Q11*Z21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) + (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + p*(Y21 - Y31)/6
        data[order[ind + 30]] += area*t*(C33*(Q31*X21 + Q33*X31)**2 + Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) + s_0 + (Q21*X21 + Q23*X31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 31]] += -Z31*p/6 + area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Y21 + Q33*Y31) + Q11*Y21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) + (Q21*Y21 + Q23*Y31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 32]] += Y31*p/6 + area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Z21 + Q33*Z31) + Q11*Z21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) + (Q21*Z21 + Q23*Z31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 33]] += area*t*(C33*Q33*X21*(Q31*X21 + Q33*X31) + s_2 + (Q22*X31 + Q23*X21)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 34]] += Z21*p/6 + area*t*(C33*Q33*Y21*(Q31*X21 + Q33*X31) + (Q22*Y31 + Q23*Y21)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 35]] += -Y21*p/6 + area*t*(C33*Q33*Z21*(Q31*X21 + Q33*X31) + (Q22*Z31 + Q23*Z21)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        data[order[ind + 36]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*Y21 + Q33*Y31) - Q11*X21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + p*(Z21 - Z31)/6
        data[order[ind + 37]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(Q31*Y21 + Q33*Y31) - Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + s2 + (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))
        data[order[ind + 38]] += area*t*(C33*(Q31*Y21 + Q33*Y31)*(-Q31*Z21 + Q33*(-Z21 - Z31)) - Q11*Z21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + p*(-X21 + X31)/6
        data[order[ind + 39]] += Z31*p/6 + area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Y21 + Q33*Y31) + Q11*X21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + (Q21*X21 + Q23*X31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 40]] += area*t*(C33*(Q31*Y21 + Q33*Y31)**2 + Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + s_0 + (Q21*Y21 + Q23*Y31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 41]] += -X31*p/6 + area*t*(C33*(Q31*Y21 + Q33*Y31)*(Q31*Z21 + Q33*Z31) + Q11*Z21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + (Q21*Z21 + Q23*Z31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 42]] += -Z21*p/6 + area*t*(C33*Q33*X21*(Q31*Y21 + Q33*Y31) + (Q22*X31 + Q23*X21)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 43]] += area*t*(C33*Q33*Y21*(Q31*Y21 + Q33*Y31) + s_2 + (Q22*Y31 + Q23*Y21)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 44]] += X21*p/6 + area*t*(C33*Q33*Z21*(Q31*Y21 + Q33*Y31) + (Q22*Z31 + Q23*Z21)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        data[order[ind + 45]] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))*(Q31*Z21 + Q33*Z31) - Q11*X21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) + (C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + p*(-Y21 + Y31)/6
        data[order[ind + 46]] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))*(Q31*Z21 + Q33*Z31) - Q11*Y21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) + (C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + p*(X21 - X31)/6
        data[order[ind + 47]] += area*t*(C33*(-Q31*Z21 + Q33*(-Z21 - Z31))*(Q31*Z21 + Q33*Z31) - Q11*Z21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) + s2 + (C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)))
        data[order[ind + 48]] += -Y31*p/6 + area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Z21 + Q33*Z31) + Q11*X21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) + (Q21*X21 + Q23*X31)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
        data[order[ind + 49]] += X31*p/6 + area*t*(C33*(Q31*Y21 + Q33*Y31)*(Q31*Z21 + Q33*Z31) + Q11*Y21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) + (Q21*Y21 + Q23*Y31)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
        data[order[ind + 50]] += area*t*(C33*(Q31*Z21 + Q33*Z31)**2 + Q11*Z21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) + s_0 + (Q21*Z21 + Q23*Z31)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
        data[order[ind + 51]] += Y21*p/6 + area*t*(C33*Q33*X21*(Q31*Z21 + Q33*Z31) + (Q22*X31 + Q23*X21)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
        data[order[ind + 52]] += -X21*p/6 + area*t*(C33*Q33*Y21*(Q31*Z21 + Q33*Z31) + (Q22*Y31 + Q23*Y21)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
        data[order[ind + 53]] += area*t*(C33*Q33*Z21*(Q31*Z21 + Q33*Z31) + s_2 + (Q22*Z31 + Q23*Z21)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
        data[order[ind + 54]] += area*t*(-C12*Q11*X21*(Q22*X31 + Q23*X21) + C22*(Q22*X31 + Q23*X21)*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + C33*Q33*X21*(-Q31*X21 + Q33*(-X21 - X31)) + s3)
        data[order[ind + 55]] += area*t*(-C12*Q11*Y21*(Q22*X31 + Q23*X21) + C22*(Q22*X31 + Q23*X21)*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + C33*Q33*X21*(-Q31*Y21 + Q33*(-Y21 - Y31))) + p*(-Z21 + Z31)/6
        data[order[ind + 56]] += area*t*(-C12*Q11*Z21*(Q22*X31 + Q23*X21) + C22*(Q22*X31 + Q23*X21)*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)) + C33*Q33*X21*(-Q31*Z21 + Q33*(-Z21 - Z31))) + p*(Y21 - Y31)/6
        data[order[ind + 57]] += area*t*(C12*Q11*X21*(Q22*X31 + Q23*X21) + C22*(Q21*X21 + Q23*X31)*(Q22*X31 + Q23*X21) + C33*Q33*X21*(Q31*X21 + Q33*X31) + s_2)
        data[order[ind + 58]] += -Z31*p/6 + area*t*(C12*Q11*Y21*(Q22*X31 + Q23*X21) + C22*(Q21*Y21 + Q23*Y31)*(Q22*X31 + Q23*X21) + C33*Q33*X21*(Q31*Y21 + Q33*Y31))
        data[order[ind + 59]] += Y31*p/6 + area*t*(C12*Q11*Z21*(Q22*X31 + Q23*X21) + C22*(Q21*Z21 + Q23*Z31)*(Q22*X31 + Q23*X21) + C33*Q33*X21*(Q31*Z21 + Q33*Z31))
        data[order[ind + 60]] += area*t*(C22*(Q22*X31 + Q23*X21)**2 + C33*Q33**2*X21**2 + s_1)
        data[order[ind + 61]] += Z21*p/6 + area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Y31 + Q23*Y21) + C33*Q33**2*X21*Y21)
        data[order[ind + 62]] += -Y21*p/6 + area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Z31 + Q23*Z21) + C33*Q33**2*X21*Z21)
        data[order[ind + 63]] += area*t*(-C12*Q11*X21*(Q22*Y31 + Q23*Y21) + C22*(Q22*Y31 + Q23*Y21)*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + C33*Q33*Y21*(-Q31*X21 + Q33*(-X21 - X31))) + p*(Z21 - Z31)/6
        data[order[ind + 64]] += area*t*(-C12*Q11*Y21*(Q22*Y31 + Q23*Y21) + C22*(Q22*Y31 + Q23*Y21)*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + C33*Q33*Y21*(-Q31*Y21 + Q33*(-Y21 - Y31)) + s3)
        data[order[ind + 65]] += area*t*(-C12*Q11*Z21*(Q22*Y31 + Q23*Y21) + C22*(Q22*Y31 + Q23*Y21)*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)) + C33*Q33*Y21*(-Q31*Z21 + Q33*(-Z21 - Z31))) + p*(-X21 + X31)/6
        data[order[ind + 66]] += Z31*p/6 + area*t*(C12*Q11*X21*(Q22*Y31 + Q23*Y21) + C22*(Q21*X21 + Q23*X31)*(Q22*Y31 + Q23*Y21) + C33*Q33*Y21*(Q31*X21 + Q33*X31))
        data[order[ind + 67]] += area*t*(C12*Q11*Y21*(Q22*Y31 + Q23*Y21) + C22*(Q21*Y21 + Q23*Y31)*(Q22*Y31 + Q23*Y21) + C33*Q33*Y21*(Q31*Y21 + Q33*Y31) + s_2)
        data[order[ind + 68]] += -X31*p/6 + area*t*(C12*Q11*Z21*(Q22*Y31 + Q23*Y21) + C22*(Q21*Z21 + Q23*Z31)*(Q22*Y31 + Q23*Y21) + C33*Q33*Y21*(Q31*Z21 + Q33*Z31))
        data[order[ind + 69]] += -Z21*p/6 + area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Y31 + Q23*Y21) + C33*Q33**2*X21*Y21)
        data[order[ind + 70]] += area*t*(C22*(Q22*Y31 + Q23*Y21)**2 + C33*Q33**2*Y21**2 + s_1)
        data[order[ind + 71]] += X21*p/6 + area*t*(C22*(Q22*Y31 + Q23*Y21)*(Q22*Z31 + Q23*Z21) + C33*Q33**2*Y21*Z21)
        data[order[ind + 72]] += area*t*(-C12*Q11*X21*(Q22*Z31 + Q23*Z21) + C22*(Q22*Z31 + Q23*Z21)*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + C33*Q33*Z21*(-Q31*X21 + Q33*(-X21 - X31))) + p*(-Y21 + Y31)/6
        data[order[ind + 73]] += area*t*(-C12*Q11*Y21*(Q22*Z31 + Q23*Z21) + C22*(Q22*Z31 + Q23*Z21)*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + C33*Q33*Z21*(-Q31*Y21 + Q33*(-Y21 - Y31))) + p*(X21 - X31)/6
        data[order[ind + 74]] += area*t*(-C12*Q11*Z21*(Q22*Z31 + Q23*Z21) + C22*(Q22*Z31 + Q23*Z21)*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)) + C33*Q33*Z21*(-Q31*Z21 + Q33*(-Z21 - Z31)) + s3)
        data[order[ind + 75]] += -Y31*p/6 + area*t*(C12*Q11*X21*(Q22*Z31 + Q23*Z21) + C22*(Q21*X21 + Q23*X31)*(Q22*Z31 + Q23*Z21) + C33*Q33*Z21*(Q31*X21 + Q33*X31))
        data[order[ind + 76]] += X31*p/6 + area*t*(C12*Q11*Y21*(Q22*Z31 + Q23*Z21) + C22*(Q21*Y21 + Q23*Y31)*(Q22*Z31 + Q23*Z21) + C33*Q33*Z21*(Q31*Y21 + Q33*Y31))
        data[order[ind + 77]] += area*t*(C12*Q11*Z21*(Q22*Z31 + Q23*Z21) + C22*(Q21*Z21 + Q23*Z31)*(Q22*Z31 + Q23*Z21) + C33*Q33*Z21*(Q31*Z21 + Q33*Z31) + s_2)
        data[order[ind + 78]] += Y21*p/6 + area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Z31 + Q23*Z21) + C33*Q33**2*X21*Z21)
        data[order[ind + 79]] += -X21*p/6 + area*t*(C22*(Q22*Y31 + Q23*Y21)*(Q22*Z31 + Q23*Z21) + C33*Q33**2*Y21*Z21)
        data[order[ind + 80]] += area*t*(C22*(Q22*Z31 + Q23*Z21)**2 + C33*Q33**2*Z21**2 + s_1)

        diagK[dof_1] += area*t*(C33*(-Q31*X21 + Q33*(-X21 - X31))**2 - Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))) + s1 + (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)))
        diagK[dof_2] += area*t*(C33*(-Q31*Y21 + Q33*(-Y21 - Y31))**2 - Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))) + s1 + (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)))
        diagK[dof_3] += area*t*(C33*(-Q31*Z21 + Q33*(-Z21 - Z31))**2 - Q11*Z21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))) + s1 + (-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)))*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)))
        diagK[dof_4] += area*t*(C33*(Q31*X21 + Q33*X31)**2 + Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) + s_0 + (Q21*X21 + Q23*X31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
        diagK[dof_5] += area*t*(C33*(Q31*Y21 + Q33*Y31)**2 + Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) + s_0 + (Q21*Y21 + Q23*Y31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
        diagK[dof_6] += area*t*(C33*(Q31*Z21 + Q33*Z31)**2 + Q11*Z21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) + s_0 + (Q21*Z21 + Q23*Z31)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
        diagK[dof_7] += area*t*(C22*(Q22*X31 + Q23*X21)**2 + C33*Q33**2*X21**2 + s_1)
        diagK[dof_8] += area*t*(C22*(Q22*Y31 + Q23*Y21)**2 + C33*Q33**2*Y21**2 + s_1)
        diagK[dof_9] += area*t*(C22*(Q22*Z31 + Q23*Z21)**2 + C33*Q33**2*Z21**2 + s_1)


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
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
                                          double beta) nogil:

    cdef:

        double Q11 = 1 / (J11 * J11)
        double Q21 = (J12 * J12) / (J11 * J11 * J22 * J22)
        double Q31 = - 2 * J12 / (J11 * J11 * J22)
        double Q22 = 1 / (J22 * J22)
        double Q23 = - J12 / (J11 * J22 * J22)
        double Q33 = 1 / (J11 * J22)

        double X21, X31, Y21, Y31, Z21, Z31

        Py_ssize_t dof_1, dof_2, dof_3, dof_4, dof_5, dof_6, dof_7, dof_8, dof_9
        
    with cython.boundscheck(False):

        X21 = X[N[el, 2]] - X[N[el, 1]]
        X31 = X[N[el, 3]] - X[N[el, 1]]
        Y21 = Y[N[el, 2]] - Y[N[el, 1]]
        Y31 = Y[N[el, 3]] - Y[N[el, 1]]
        Z21 = Z[N[el, 2]] - Z[N[el, 1]]
        Z31 = Z[N[el, 3]] - Z[N[el, 1]]

        # Find degrees of freedom from current element
        dof_1 = 3 * (N[el, 1] + 1) - 3
        dof_2 = 3 * (N[el, 1] + 1) - 2
        dof_3 = 3 * (N[el, 1] + 1) - 1
        dof_4 = 3 * (N[el, 2] + 1) - 3
        dof_5 = 3 * (N[el, 2] + 1) - 2
        dof_6 = 3 * (N[el, 2] + 1) - 1
        dof_7 = 3 * (N[el, 3] + 1) - 3
        dof_8 = 3 * (N[el, 3] + 1) - 2
        dof_9 = 3 * (N[el, 3] + 1) - 1

        # internal force vector
        Fint[dof_1] += area*t*(-Q11*SLocal[0]*X21 + SLocal[1]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + SLocal[2]*(-Q31*X21 + Q33*(-X21 - X31)))
        Fint[dof_2] += area*t*(-Q11*SLocal[0]*Y21 + SLocal[1]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + SLocal[2]*(-Q31*Y21 + Q33*(-Y21 - Y31)))
        Fint[dof_3] += area*t*(-Q11*SLocal[0]*Z21 + SLocal[1]*(-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31)) + SLocal[2]*(-Q31*Z21 + Q33*(-Z21 - Z31)))
        Fint[dof_4] += area*t*(Q11*SLocal[0]*X21 + SLocal[1]*(Q21*X21 + Q23*X31) + SLocal[2]*(Q31*X21 + Q33*X31))
        Fint[dof_5] += area*t*(Q11*SLocal[0]*Y21 + SLocal[1]*(Q21*Y21 + Q23*Y31) + SLocal[2]*(Q31*Y21 + Q33*Y31))
        Fint[dof_6] += area*t*(Q11*SLocal[0]*Z21 + SLocal[1]*(Q21*Z21 + Q23*Z31) + SLocal[2]*(Q31*Z21 + Q33*Z31))
        Fint[dof_7] += area*t*(Q33*SLocal[2]*X21 + SLocal[1]*(Q22*X31 + Q23*X21))
        Fint[dof_8] += area*t*(Q33*SLocal[2]*Y21 + SLocal[1]*(Q22*Y31 + Q23*Y21))
        Fint[dof_9] += area*t*(Q33*SLocal[2]*Z21 + SLocal[1]*(Q22*Z31 + Q23*Z21))


# @cython.profile(False)
# @cython.cdivision(True)
# @cython.wraparound(False)
# @cython.nonecheck(False)
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
#                             double tau) nogil:

#     cdef:
#         long double Q11 = 1 / (J11 * J11)
#         long double Q21 = (J12 * J12) / (J11 * J11 * J22 * J22)
#         long double Q31 = - 2 * J12 / (J11 * J11 * J22)
#         long double Q22 = 1 / (J22 * J22)
#         long double Q23 = - J12 / (J11 * J22 * J22)
#         long double Q33 = 1 / (J11 * J22)

#         double s_0 = Q11 * SLocal[0] + Q21 * SLocal[1] + Q31 * SLocal[2]
#         double s_1 = Q22 * SLocal[1]
#         double s_2 = Q23 * SLocal[1] + Q33 * SLocal[2]

#         double s1 = (s_0 + s_1 + 2 * s_2) * area * t
#         double s2 = - (s_0 + s_2) * area * t
#         double s3 = - (s_1 + s_2) * area * t
#         double s00 = s_0 * area * t
#         double s11 = s_1 * area * t
#         double s22 = s_2 * area * t

#         double X23 = (X[N[el, 2]] - X[N[el, 3]]) * p / 6
#         double X13 = (X[N[el, 1]] - X[N[el, 3]]) * p / 6
#         double X12 = (X[N[el, 1]] - X[N[el, 2]]) * p / 6

#         double Y23 = (Y[N[el, 2]] - Y[N[el, 3]]) * p / 6
#         double Y13 = (Y[N[el, 1]] - Y[N[el, 3]]) * p / 6
#         double Y12 = (Y[N[el, 1]] - Y[N[el, 2]]) * p / 6

#         double Z23 = (Z[N[el, 2]] - Z[N[el, 3]]) * p / 6
#         double Z13 = (Z[N[el, 1]] - Z[N[el, 3]]) * p / 6
#         double Z12 = (Z[N[el, 1]] - Z[N[el, 2]]) * p / 6

#         double X21 = X[N[el, 2]] - X[N[el, 1]]
#         double X31 = X[N[el, 3]] - X[N[el, 1]]

#         double Y21 = Y[N[el, 2]] - Y[N[el, 1]]
#         double Y31 = Y[N[el, 3]] - Y[N[el, 1]]

#         double Z21 = Z[N[el, 2]] - Z[N[el, 1]]
#         double Z31 = Z[N[el, 3]] - Z[N[el, 1]]

#         # local constitutive matrix parameters
#         double C11 = E / (1 - nu * nu)
#         double C12 = nu * E / (1 - nu * nu)
#         double C22 = E / (1 - nu * nu)
#         double C33 = 0.5 * (1 - nu) * E / (1 - nu * nu)

#         double V0 = V[dof_1] * tau
#         double V1 = V[dof_2] * tau
#         double V2 = V[dof_3] * tau
#         double V3 = V[dof_4] * tau
#         double V4 = V[dof_5] * tau
#         double V5 = V[dof_6] * tau
#         double V6 = V[dof_7] * tau
#         double V7 = V[dof_8] * tau
#         double V8 = V[dof_9] * tau

#         Py_ssize_t dof_1, dof_2, dof_3, dof_4, dof_5, dof_6, dof_7, dof_8, dof_9
        
#     # Find degrees of freedom from current element
#     dof_1 = 3 * (N[el, 1] + 1) - 3
#     dof_2 = 3 * (N[el, 1] + 1) - 2
#     dof_3 = 3 * (N[el, 1] + 1) - 1
#     dof_4 = 3 * (N[el, 2] + 1) - 3
#     dof_5 = 3 * (N[el, 2] + 1) - 2
#     dof_6 = 3 * (N[el, 2] + 1) - 1
#     dof_7 = 3 * (N[el, 3] + 1) - 3
#     dof_8 = 3 * (N[el, 3] + 1) - 2
#     dof_9 = 3 * (N[el, 3] + 1) - 1

#     data[order[ind]] += area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(-Q31*X21 - Q33*(X21 + X31)) -
#                          Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) +
#                          (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))*
#                          (-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) + s1
#     data[order[ind + 1]] += -Z23 + area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(-Q31*Y21 - Q33*(Y21 + Y31)) -
#                                 Q11*Y21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) +
#                                 (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))*
#                                 (-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))
#     data[order[ind + 2]] += Y23 + area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(-Q31*Z21 - Q33*(Z21 + Z31)) -
#                                Q11*Z21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) +
#                                (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))*
#                                (-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))
#     data[order[ind + 3]] += area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(Q31*X21 + Q33*X31) +
#                          Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) +
#                          (Q21*X21 + Q23*X31)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))) + s2
#     data[order[ind + 4]] += Z13 + area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(Q31*Y21 + Q33*Y31) +
#                                Q11*Y21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) +
#                                (Q21*Y21 + Q23*Y31)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))))
#     data[order[ind + 5]] += -Y13 + area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(Q31*Z21 + Q33*Z31) +
#                                 Q11*Z21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) +
#                                 (Q21*Z21 + Q23*Z31)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))))
#     data[order[ind + 6]] += area*t*(C33*Q33*X21*(-Q31*X21 - Q33*(X21 + X31)) +
#                          (Q22*X31 + Q23*X21)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))) + s3
#     data[order[ind + 7]] += -Z12 + area*t*(C33*Q33*Y21*(-Q31*X21 - Q33*(X21 + X31)) +
#                                 (Q22*Y31 + Q23*Y21)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))))
#     data[order[ind + 8]] += Y12 + area*t*(C33*Q33*Z21*(-Q31*X21 - Q33*(X21 + X31)) +
#                                (Q22*Z31 + Q23*Z21)*(-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))))

#     data[order[ind + 9]] += Z23 + area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(-Q31*Y21 - Q33*(Y21 + Y31)) -
#                                Q11*X21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) +
#                                (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))*
#                                (-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))
#     data[order[ind + 10]] += area*t*(C33*(-Q31*Y21 - Q33*(Y21 + Y31))*(-Q31*Y21 - Q33*(Y21 + Y31)) -
#                          Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) +
#                          (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))*
#                          (-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) + s1
#     data[order[ind + 11]] += -X23 + area*t*(C33*(-Q31*Y21 - Q33*(Y21 + Y31))*(-Q31*Z21 - Q33*(Z21 + Z31)) -
#                                 Q11*Z21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) +
#                                 (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))*
#                                 (-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))
#     data[order[ind + 12]] += -Z13 + area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Y21 - Q33*(Y21 + Y31)) +
#                                 Q11*X21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) +
#                                 (Q21*X21 + Q23*X31)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))))
#     data[order[ind + 13]] += area*t*(C33*(-Q31*Y21 - Q33*(Y21 + Y31))*(Q31*Y21 + Q33*Y31) +
#                          Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) +
#                          (Q21*Y21 + Q23*Y31)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))) + s2
#     data[order[ind + 14]] += X13 + area*t*(C33*(-Q31*Y21 - Q33*(Y21 + Y31))*(Q31*Z21 + Q33*Z31) +
#                                Q11*Z21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) +
#                                (Q21*Z21 + Q23*Z31)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))))
#     data[order[ind + 15]] += Z12 + area*t*(C33*Q33*X21*(-Q31*Y21 - Q33*(Y21 + Y31)) +
#                                (Q22*X31 + Q23*X21)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))))
#     data[order[ind + 16]] += area*t*(C33*Q33*Y21*(-Q31*Y21 - Q33*(Y21 + Y31)) +
#                          (Q22*Y31 + Q23*Y21)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))) + s3
#     data[order[ind + 17]] += -X12 + area*t*(C33*Q33*Z21*(-Q31*Y21 - Q33*(Y21 + Y31)) +
#                                 (Q22*Z31 + Q23*Z21)*(-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))))

#     data[order[ind + 18]] += -Y23 + area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(-Q31*Z21 - Q33*(Z21 + Z31)) -
#                                 Q11*X21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) +
#                                 (-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))*
#                                 (-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))
#     data[order[ind + 19]] += X23 + area*t*(C33*(-Q31*Y21 - Q33*(Y21 + Y31))*(-Q31*Z21 - Q33*(Z21 + Z31)) -
#                                Q11*Y21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) +
#                                (-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))*
#                                (-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))
#     data[order[ind + 20]] += area*t*(C33*(-Q31*Z21 - Q33*(Z21 + Z31))*(-Q31*Z21 - Q33*(Z21 + Z31)) -
#                          Q11*Z21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) +
#                          (-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))*
#                          (-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) + s1
#     data[order[ind + 21]] += Y13 + area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Z21 - Q33*(Z21 + Z31)) +
#                                Q11*X21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) +
#                                (Q21*X21 + Q23*X31)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))))
#     data[order[ind + 22]] += -X13 + area*t*(C33*(Q31*Y21 + Q33*Y31)*(-Q31*Z21 - Q33*(Z21 + Z31)) +
#                                 Q11*Y21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) +
#                                 (Q21*Y21 + Q23*Y31)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))))
#     data[order[ind + 23]] += area*t*(C33*(-Q31*Z21 - Q33*(Z21 + Z31))*(Q31*Z21 + Q33*Z31) +
#                          Q11*Z21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) +
#                          (Q21*Z21 + Q23*Z31)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))) + s2
#     data[order[ind + 24]] += -Y12 + area*t*(C33*Q33*X21*(-Q31*Z21 - Q33*(Z21 + Z31)) +
#                                 (Q22*X31 + Q23*X21)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))))
#     data[order[ind + 25]] += X12 + area*t*(C33*Q33*Y21*(-Q31*Z21 - Q33*(Z21 + Z31)) +
#                                (Q22*Y31 + Q23*Y21)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))))
#     data[order[ind + 26]] += area*t*(C33*Q33*Z21*(-Q31*Z21 - Q33*(Z21 + Z31)) +
#                          (Q22*Z31 + Q23*Z21)*(-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))) + s3

#     data[order[ind + 27]] += area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(Q31*X21 + Q33*X31) -
#                          Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
#                          (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) + s2
#     data[order[ind + 28]] += -Z23 + area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Y21 - Q33*(Y21 + Y31)) -
#                                 Q11*Y21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
#                                 (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))
#     data[order[ind + 29]] += Y23 + area*t*(C33*(Q31*X21 + Q33*X31)*(-Q31*Z21 - Q33*(Z21 + Z31)) -
#                                Q11*Z21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
#                                (C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))
#     data[order[ind + 30]] += area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*X21 + Q33*X31) +
#                          Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
#                          (Q21*X21 + Q23*X31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))) + s00
#     data[order[ind + 31]] += Z13 + area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Y21 + Q33*Y31) +
#                                Q11*Y21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
#                                (Q21*Y21 + Q23*Y31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
#     data[order[ind + 32]] += -Y13 + area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Z21 + Q33*Z31) +
#                                 Q11*Z21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
#                                 (Q21*Z21 + Q23*Z31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
#     data[order[ind + 33]] += area*t*(C33*Q33*X21*(Q31*X21 + Q33*X31) +
#                          (Q22*X31 + Q23*X21)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))) + s22
#     data[order[ind + 34]] += -Z12 + area*t*(C33*Q33*Y21*(Q31*X21 + Q33*X31) +
#                                 (Q22*Y31 + Q23*Y21)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))
#     data[order[ind + 35]] += Y12 + area*t*(C33*Q33*Z21*(Q31*X21 + Q33*X31) +
#                                (Q22*Z31 + Q23*Z21)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31)))

#     data[order[ind + 36]] += Z23 + area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(Q31*Y21 + Q33*Y31) -
#                                Q11*X21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
#                                (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))
#     data[order[ind + 37]] += area*t*(C33*(-Q31*Y21 - Q33*(Y21 + Y31))*(Q31*Y21 + Q33*Y31) -
#                          Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
#                          (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) + s2
#     data[order[ind + 38]] += -X23 + area*t*(C33*(Q31*Y21 + Q33*Y31)*(-Q31*Z21 - Q33*(Z21 + Z31)) -
#                                 Q11*Z21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
#                                 (C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))
#     data[order[ind + 39]] += -Z13 + area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Y21 + Q33*Y31) +
#                                 Q11*X21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
#                                 (Q21*X21 + Q23*X31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
#     data[order[ind + 40]] += area*t*(C33*(Q31*Y21 + Q33*Y31)*(Q31*Y21 + Q33*Y31) +
#                          Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
#                          (Q21*Y21 + Q23*Y31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))) + s00
#     data[order[ind + 41]] += X13 + area*t*(C33*(Q31*Y21 + Q33*Y31)*(Q31*Z21 + Q33*Z31) +
#                                Q11*Z21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
#                                (Q21*Z21 + Q23*Z31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
#     data[order[ind + 42]] += Z12 + area*t*(C33*Q33*X21*(Q31*Y21 + Q33*Y31) +
#                                (Q22*X31 + Q23*X21)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))
#     data[order[ind + 43]] += area*t*(C33*Q33*Y21*(Q31*Y21 + Q33*Y31) +
#                          (Q22*Y31 + Q23*Y21)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))) + s22
#     data[order[ind + 44]] += -X12 + area*t*(C33*Q33*Z21*(Q31*Y21 + Q33*Y31) +
#                                 (Q22*Z31 + Q23*Z21)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31)))

#     data[order[ind + 45]] += -Y23 + area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(Q31*Z21 + Q33*Z31) -
#                                 Q11*X21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) +
#                                 (C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))
#     data[order[ind + 46]] += X23 + area*t*(C33*(-Q31*Y21 - Q33*(Y21 + Y31))*(Q31*Z21 + Q33*Z31) -
#                                Q11*Y21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) +
#                                (C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))
#     data[order[ind + 47]] += area*t*(C33*(-Q31*Z21 - Q33*(Z21 + Z31))*(Q31*Z21 + Q33*Z31) -
#                          Q11*Z21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) +
#                          (C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) + s2
#     data[order[ind + 48]]+= Y13 + area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*Z21 + Q33*Z31) +
#                                Q11*X21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) +
#                                (Q21*X21 + Q23*X31)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
#     data[order[ind + 49]] += -X13 + area*t*(C33*(Q31*Y21 + Q33*Y31)*(Q31*Z21 + Q33*Z31) +
#                                 Q11*Y21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) +
#                                 (Q21*Y21 + Q23*Y31)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
#     data[order[ind + 50]] += area*t*(C33*(Q31*Z21 + Q33*Z31)*(Q31*Z21 + Q33*Z31) +
#                          Q11*Z21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) +
#                          (Q21*Z21 + Q23*Z31)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))) + s00
#     data[order[ind + 51]] += -Y12 + area*t*(C33*Q33*X21*(Q31*Z21 + Q33*Z31) +
#                                 (Q22*X31 + Q23*X21)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
#     data[order[ind + 52]] += X12 + area*t*(C33*Q33*Y21*(Q31*Z21 + Q33*Z31) +
#                                (Q22*Y31 + Q23*Y21)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31)))
#     data[order[ind + 53]] += area*t*(C33*Q33*Z21*(Q31*Z21 + Q33*Z31) +
#                          (Q22*Z31 + Q23*Z21)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))) + s22

#     data[order[ind + 54]] += area*t*(-C12*Q11*X21*(Q22*X31 + Q23*X21) +
#                          C22*(Q22*X31 + Q23*X21)*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)) +
#                          C33*Q33*X21*(-Q31*X21 - Q33*(X21 + X31))) + s3
#     data[order[ind + 55]] += -Z23 + area*t*(-C12*Q11*Y21*(Q22*X31 + Q23*X21) +
#                                 C22*(Q22*X31 + Q23*X21)*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)) +
#                                 C33*Q33*X21*(-Q31*Y21 - Q33*(Y21 + Y31)))
#     data[order[ind + 56]] += Y23 + area*t*(-C12*Q11*Z21*(Q22*X31 + Q23*X21) +
#                                C22*(Q22*X31 + Q23*X21)*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)) +
#                                C33*Q33*X21*(-Q31*Z21 - Q33*(Z21 + Z31)))
#     data[order[ind + 57]] += area*t*(C12*Q11*X21*(Q22*X31 + Q23*X21) + C22*(Q21*X21 + Q23*X31)*(Q22*X31 + Q23*X21) +
#                          C33*Q33*X21*(Q31*X21 + Q33*X31)) + s22
#     data[order[ind + 58]] += Z13 + area*t*(C12*Q11*Y21*(Q22*X31 + Q23*X21) +
#                                            C22*(Q21*Y21 + Q23*Y31)*(Q22*X31 + Q23*X21) +
#                                            C33*Q33*X21*(Q31*Y21 + Q33*Y31))
#     data[order[ind + 59]] += -Y13 + area*t*(C12*Q11*Z21*(Q22*X31 + Q23*X21) +
#                                             C22*(Q21*Z21 + Q23*Z31)*(Q22*X31 + Q23*X21) +
#                                             C33*Q33*X21*(Q31*Z21 + Q33*Z31))
#     data[order[ind + 60]] += area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*X31 + Q23*X21) + C33*Q33*Q33*X21*X21) + s11
#     data[order[ind + 61]] += -Z12 + area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Y31 + Q23*Y21) + C33*Q33*Q33*X21*Y21)
#     data[order[ind + 62]] += Y12 + area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Z31 + Q23*Z21) + C33*Q33*Q33*X21*Z21)

#     data[order[ind + 63]] += Z23 + area*t*(-C12*Q11*X21*(Q22*Y31 + Q23*Y21) +
#                                C22*(Q22*Y31 + Q23*Y21)*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)) +
#                                C33*Q33*Y21*(-Q31*X21 - Q33*(X21 + X31)))
#     data[order[ind + 64]] += area*t*(-C12*Q11*Y21*(Q22*Y31 + Q23*Y21) +
#                          C22*(Q22*Y31 + Q23*Y21)*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)) +
#                          C33*Q33*Y21*(-Q31*Y21 - Q33*(Y21 + Y31))) + s3
#     data[order[ind + 65]] += -X23 + area*t*(-C12*Q11*Z21*(Q22*Y31 + Q23*Y21) +
#                                 C22*(Q22*Y31 + Q23*Y21)*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)) +
#                                 C33*Q33*Y21*(-Q31*Z21 - Q33*(Z21 + Z31)))
#     data[order[ind + 66]] += -Z13 + area*t*(C12*Q11*X21*(Q22*Y31 + Q23*Y21) +
#                                 C22*(Q21*X21 + Q23*X31)*(Q22*Y31 + Q23*Y21) + C33*Q33*Y21*(Q31*X21 + Q33*X31))
#     data[order[ind + 67]] += area*t*(C12*Q11*Y21*(Q22*Y31 + Q23*Y21) +
#                          C22*(Q21*Y21 + Q23*Y31)*(Q22*Y31 + Q23*Y21) + C33*Q33*Y21*(Q31*Y21 + Q33*Y31)) + s22
#     data[order[ind + 68]] += X13 + area*t*(C12*Q11*Z21*(Q22*Y31 + Q23*Y21) +
#                                C22*(Q21*Z21 + Q23*Z31)*(Q22*Y31 + Q23*Y21) + C33*Q33*Y21*(Q31*Z21 + Q33*Z31))
#     data[order[ind + 69]] += Z12 + area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Y31 + Q23*Y21) + C33*Q33*Q33*X21*Y21)
#     data[order[ind + 70]] += area*t*(C22*(Q22*Y31 + Q23*Y21)*(Q22*Y31 + Q23*Y21) + C33*Q33*Q33*Y21*Y21) + s11
#     data[order[ind + 71]] += -X12 + area*t*(C22*(Q22*Y31 + Q23*Y21)*(Q22*Z31 + Q23*Z21) + C33*Q33*Q33*Y21*Z21)

#     data[order[ind + 72]] += -Y23 + area*t*(-C12*Q11*X21*(Q22*Z31 + Q23*Z21) +
#                                 C22*(Q22*Z31 + Q23*Z21)*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)) +
#                                 C33*Q33*Z21*(-Q31*X21 - Q33*(X21 + X31)))
#     data[order[ind + 73]] += X23 + area*t*(-C12*Q11*Y21*(Q22*Z31 + Q23*Z21) +
#                                C22*(Q22*Z31 +Q23*Z21)*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)) +
#                                C33*Q33*Z21*(-Q31*Y21 - Q33*(Y21 + Y31)))
#     data[order[ind + 74]] += area*t*(-C12*Q11*Z21*(Q22*Z31 + Q23*Z21) +
#                          C22*(Q22*Z31 + Q23*Z21)*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)) +
#                          C33*Q33*Z21*(-Q31*Z21 - Q33*(Z21 + Z31))) + s3
#     data[order[ind + 75]] += Y13 + area*t*(C12*Q11*X21*(Q22*Z31 + Q23*Z21) +
#                                C22*(Q21*X21 + Q23*X31)*(Q22*Z31 + Q23*Z21) + C33*Q33*Z21*(Q31*X21 + Q33*X31))
#     data[order[ind + 76]] += -X13 + area*t*(C12*Q11*Y21*(Q22*Z31 + Q23*Z21) +
#                                 C22*(Q21*Y21 + Q23*Y31)*(Q22*Z31 + Q23*Z21) + C33*Q33*Z21*(Q31*Y21 + Q33*Y31))
#     data[order[ind + 77]] += area*t*(C12*Q11*Z21*(Q22*Z31 + Q23*Z21) +
#                          C22*(Q21*Z21 + Q23*Z31)*(Q22*Z31 + Q23*Z21) + C33*Q33*Z21*(Q31*Z21 + Q33*Z31)) + s22
#     data[order[ind + 78]] += -Y12 + area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*Z31 + Q23*Z21) + C33*Q33*Q33*X21*Z21)
#     data[order[ind + 79]] += X12 + area*t*(C22*(Q22*Y31 + Q23*Y21)*(Q22*Z31 + Q23*Z21) + C33*Q33*Q33*Y21*Z21)
#     data[order[ind + 80]] += area*t*(C22*(Q22*Z31 + Q23*Z21)*(Q22*Z31 + Q23*Z21) + C33*Q33*Q33*Z21*Z21) + s11

#     # internal force vector
#     Fint[dof_1] += (-Q11*X21*(SLocal[0] + V0*(-C12*Q22*X31 + C12*Q23*(-X21 - X31) - X21*(C11*Q11 + C12*Q21)) + V1*(-C12*Q22*Y31 + C12*Q23*(-Y21 - Y31) - Y21*(C11*Q11 + C12*Q21)) + V2*(-C12*Q22*Z31 + C12*Q23*(-Z21 - Z31) - Z21*(C11*Q11 + C12*Q21)) + V3*(C12*Q23*X31 + X21*(C11*Q11 + C12*Q21)) + V4*(C12*Q23*Y31 + Y21*(C11*Q11 + C12*Q21)) + V5*(C12*Q23*Z31 + Z21*(C11*Q11 + C12*Q21)) + V6*(C12*Q22*X31 + C12*Q23*X21) + V7*(C12*Q22*Y31 + C12*Q23*Y21) + V8*(C12*Q22*Z31 + C12*Q23*Z21)) + (-Q31*X21 + Q33*(-X21 - X31))*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31)) + (-Q21*X21 - Q22*X31 + Q23*(-X21 - X31))*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21))) * area * t

#     Fint[dof_2] += (-Q11*Y21*(SLocal[0] + V0*(-C12*Q22*X31 + C12*Q23*(-X21 - X31) - X21*(C11*Q11 + C12*Q21)) + V1*(-C12*Q22*Y31 + C12*Q23*(-Y21 - Y31) - Y21*(C11*Q11 + C12*Q21)) + V2*(-C12*Q22*Z31 + C12*Q23*(-Z21 - Z31) - Z21*(C11*Q11 + C12*Q21)) + V3*(C12*Q23*X31 + X21*(C11*Q11 + C12*Q21)) + V4*(C12*Q23*Y31 + Y21*(C11*Q11 + C12*Q21)) + V5*(C12*Q23*Z31 + Z21*(C11*Q11 + C12*Q21)) + V6*(C12*Q22*X31 + C12*Q23*X21) + V7*(C12*Q22*Y31 + C12*Q23*Y21) + V8*(C12*Q22*Z31 + C12*Q23*Z21)) + (-Q31*Y21 + Q33*(-Y21 - Y31))*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31)) + (-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31))*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21))) * area * t

#     Fint[dof_3] += (-Q11*Z21*(SLocal[0] + V0*(-C12*Q22*X31 + C12*Q23*(-X21 - X31) - X21*(C11*Q11 + C12*Q21)) + V1*(-C12*Q22*Y31 + C12*Q23*(-Y21 - Y31) - Y21*(C11*Q11 + C12*Q21)) + V2*(-C12*Q22*Z31 + C12*Q23*(-Z21 - Z31) - Z21*(C11*Q11 + C12*Q21)) + V3*(C12*Q23*X31 + X21*(C11*Q11 + C12*Q21)) + V4*(C12*Q23*Y31 + Y21*(C11*Q11 + C12*Q21)) + V5*(C12*Q23*Z31 + Z21*(C11*Q11 + C12*Q21)) + V6*(C12*Q22*X31 + C12*Q23*X21) + V7*(C12*Q22*Y31 + C12*Q23*Y21) + V8*(C12*Q22*Z31 + C12*Q23*Z21)) + (-Q31*Z21 + Q33*(-Z21 - Z31))*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31)) + (-Q21*Z21 - Q22*Z31 + Q23*(-Z21 - Z31))*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21))) * area * t

#     Fint[dof_4] += (Q11*X21*(SLocal[0] + V0*(-C12*Q22*X31 + C12*Q23*(-X21 - X31) - X21*(C11*Q11 + C12*Q21)) + V1*(-C12*Q22*Y31 + C12*Q23*(-Y21 - Y31) - Y21*(C11*Q11 + C12*Q21)) + V2*(-C12*Q22*Z31 + C12*Q23*(-Z21 - Z31) - Z21*(C11*Q11 + C12*Q21)) + V3*(C12*Q23*X31 + X21*(C11*Q11 + C12*Q21)) + V4*(C12*Q23*Y31 + Y21*(C11*Q11 + C12*Q21)) + V5*(C12*Q23*Z31 + Z21*(C11*Q11 + C12*Q21)) + V6*(C12*Q22*X31 + C12*Q23*X21) + V7*(C12*Q22*Y31 + C12*Q23*Y21) + V8*(C12*Q22*Z31 + C12*Q23*Z21)) + (Q21*X21 + Q23*X31)*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21)) + (Q31*X21 + Q33*X31)*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31))) * area * t

#     Fint[dof_5] += (Q11*Y21*(SLocal[0] + V0*(-C12*Q22*X31 + C12*Q23*(-X21 - X31) - X21*(C11*Q11 + C12*Q21)) + V1*(-C12*Q22*Y31 + C12*Q23*(-Y21 - Y31) - Y21*(C11*Q11 + C12*Q21)) + V2*(-C12*Q22*Z31 + C12*Q23*(-Z21 - Z31) - Z21*(C11*Q11 + C12*Q21)) + V3*(C12*Q23*X31 + X21*(C11*Q11 + C12*Q21)) + V4*(C12*Q23*Y31 + Y21*(C11*Q11 + C12*Q21)) + V5*(C12*Q23*Z31 + Z21*(C11*Q11 + C12*Q21)) + V6*(C12*Q22*X31 + C12*Q23*X21) + V7*(C12*Q22*Y31 + C12*Q23*Y21) + V8*(C12*Q22*Z31 + C12*Q23*Z21)) + (Q21*Y21 + Q23*Y31)*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21)) + (Q31*Y21 + Q33*Y31)*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31))) * area * t

#     Fint[dof_6] += (Q11*Z21*(SLocal[0] + V0*(-C12*Q22*X31 + C12*Q23*(-X21 - X31) - X21*(C11*Q11 + C12*Q21)) + V1*(-C12*Q22*Y31 + C12*Q23*(-Y21 - Y31) - Y21*(C11*Q11 + C12*Q21)) + V2*(-C12*Q22*Z31 + C12*Q23*(-Z21 - Z31) - Z21*(C11*Q11 + C12*Q21)) + V3*(C12*Q23*X31 + X21*(C11*Q11 + C12*Q21)) + V4*(C12*Q23*Y31 + Y21*(C11*Q11 + C12*Q21)) + V5*(C12*Q23*Z31 + Z21*(C11*Q11 + C12*Q21)) + V6*(C12*Q22*X31 + C12*Q23*X21) + V7*(C12*Q22*Y31 + C12*Q23*Y21) + V8*(C12*Q22*Z31 + C12*Q23*Z21)) + (Q21*Z21 + Q23*Z31)*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21)) + (Q31*Z21 + Q33*Z31)*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31))) * area * t

#     Fint[dof_7] += (Q33*X21*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31)) + (Q22*X31 + Q23*X21)*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21))) * area * t

#     Fint[dof_8] += (Q33*Y21*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31)) + (Q22*Y31 + Q23*Y21)*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21))) * area * t

#     Fint[dof_9] += (Q33*Z21*(C33*Q33*V6*X21 + C33*Q33*V7*Y21 + C33*Q33*V8*Z21 + SLocal[2] + V0*(-C33*Q31*X21 + C33*Q33*(-X21 - X31)) + V1*(-C33*Q31*Y21 + C33*Q33*(-Y21 - Y31)) + V2*(-C33*Q31*Z21 + C33*Q33*(-Z21 - Z31)) + V3*(C33*Q31*X21 + C33*Q33*X31) + V4*(C33*Q31*Y21 + C33*Q33*Y31) + V5*(C33*Q31*Z21 + C33*Q33*Z31)) + (Q22*Z31 + Q23*Z21)*(SLocal[1] + V0*(-C22*Q22*X31 + C22*Q23*(-X21 - X31) - X21*(C12*Q11 + C22*Q21)) + V1*(-C22*Q22*Y31 + C22*Q23*(-Y21 - Y31) - Y21*(C12*Q11 + C22*Q21)) + V2*(-C22*Q22*Z31 + C22*Q23*(-Z21 - Z31) - Z21*(C12*Q11 + C22*Q21)) + V3*(C22*Q23*X31 + X21*(C12*Q11 + C22*Q21)) + V4*(C22*Q23*Y31 + Y21*(C12*Q11 + C22*Q21)) + V5*(C22*Q23*Z31 + Z21*(C12*Q11 + C22*Q21)) + V6*(C22*Q22*X31 + C22*Q23*X21) + V7*(C22*Q22*Y31 + C22*Q23*Y21) + V8*(C22*Q22*Z31 + C22*Q23*Z21))) * area * t

#     # diagonal
#     diagK[dof_1] += area*t*(C33*(-Q31*X21 - Q33*(X21 + X31))*(-Q31*X21 - Q33*(X21 + X31)) -
#                          Q11*X21*(-C11*Q11*X21 + C12*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) +
#                          (-C12*Q11*X21 + C22*(-Q21*X21 - Q22*X31 - Q23*(X21 + X31)))*
#                          (-Q21*X21 - Q22*X31 - Q23*(X21 + X31))) + s1
#     diagK[dof_2] += area*t*(C33*(-Q31*Y21 - Q33*(Y21 + Y31))*(-Q31*Y21 - Q33*(Y21 + Y31)) -
#                          Q11*Y21*(-C11*Q11*Y21 + C12*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) +
#                          (-C12*Q11*Y21 + C22*(-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31)))*
#                          (-Q21*Y21 - Q22*Y31 - Q23*(Y21 + Y31))) + s1
#     diagK[dof_3] += area*t*(C33*(-Q31*Z21 - Q33*(Z21 + Z31))*(-Q31*Z21 - Q33*(Z21 + Z31)) -
#                          Q11*Z21*(-C11*Q11*Z21 + C12*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) +
#                          (-C12*Q11*Z21 + C22*(-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31)))*
#                          (-Q21*Z21 - Q22*Z31 - Q23*(Z21 + Z31))) + s1
#     diagK[dof_4] += area*t*(C33*(Q31*X21 + Q33*X31)*(Q31*X21 + Q33*X31) +
#                          Q11*X21*(C11*Q11*X21 + C12*(Q21*X21 + Q23*X31)) +
#                          (Q21*X21 + Q23*X31)*(C12*Q11*X21 + C22*(Q21*X21 + Q23*X31))) + s00
#     diagK[dof_5] += area*t*(C33*(Q31*Y21 + Q33*Y31)*(Q31*Y21 + Q33*Y31) +
#                          Q11*Y21*(C11*Q11*Y21 + C12*(Q21*Y21 + Q23*Y31)) +
#                          (Q21*Y21 + Q23*Y31)*(C12*Q11*Y21 + C22*(Q21*Y21 + Q23*Y31))) + s00
#     diagK[dof_6] += area*t*(C33*(Q31*Z21 + Q33*Z31)*(Q31*Z21 + Q33*Z31) +
#                          Q11*Z21*(C11*Q11*Z21 + C12*(Q21*Z21 + Q23*Z31)) +
#                          (Q21*Z21 + Q23*Z31)*(C12*Q11*Z21 + C22*(Q21*Z21 + Q23*Z31))) + s00
#     diagK[dof_7] += area*t*(C22*(Q22*X31 + Q23*X21)*(Q22*X31 + Q23*X21) + C33*Q33*Q33*X21*X21) + s11
#     diagK[dof_8] += area*t*(C22*(Q22*Y31 + Q23*Y21)*(Q22*Y31 + Q23*Y21) + C33*Q33*Q33*Y21*Y21) + s11
#     diagK[dof_9] += area*t*(C22*(Q22*Z31 + Q23*Z21)*(Q22*Z31 + Q23*Z21) + C33*Q33*Q33*Z21*Z21) + s11


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int membraneStress(double [:] strainVoigt,
                        double [:] stressVoigt,
                        double * S1,
                        double * S2,
                        double * theta,
                        double E,
                        double nu) nogil:

    cdef:
        double a, b

    with cython.boundscheck(False):

        # determine stress in local coordinate system
        stressVoigt[0] = E / (1 - nu * nu) * (strainVoigt[0] + nu * strainVoigt[1])
        stressVoigt[1] = E / (1 - nu * nu) * (strainVoigt[1] + nu * strainVoigt[0])
        stressVoigt[2] = E / (1 - nu * nu) * (1 - nu) * 0.5 * strainVoigt[2]

        # principal strain direction from elastic strain
        theta[0] = 0.5 * atan2(strainVoigt[2], (strainVoigt[0] - strainVoigt[1]))

        # principal stress
        a = (stressVoigt[0] + stressVoigt[1]) / 2
        b = stressVoigt[2] * stressVoigt[2] - stressVoigt[0] * stressVoigt[1]

        S1[0] = a + sqrt(a * a + b)
        S2[0] = a - sqrt(a * a + b)


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
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
                                          double sigma_max) nogil:

    cdef:
        double a, b, E1, E2

        unsigned int nIter, maxIter = 20

        double s = sin(theta)
        double c = cos(theta)

        double c2 = c * c
        double s2 = s * s

        double C11 = E / (1 - nu * nu)
        double C22 = E / (1 - nu * nu)
        double C12 = E / (1 - nu * nu) * nu
        double C33 = E / (1 - nu * nu) * 0.5 * (1 - nu)

        double C_mod_11 = C11
        double C_mod_12 = C12
        double C_mod_13 = 0

        double C_mod_21 = C12
        double C_mod_22 = C22
        double C_mod_23 = 0

        double C_mod_31 = 0
        double C_mod_32 = 0
        double C_mod_33 = C33

        double S11, S22, S12, E11, E22, E12

        double f2=1, df2, dalpha, S1_hat_tilde, alpha

    with cython.boundscheck(False):

        S11 = SLocal[0]
        S22 = SLocal[1]
        S12 = SLocal[2]

        E11 = ELocal[0]
        E22 = ELocal[1]
        E12 = ELocal[2]

        # determine principal strain
        a = (E11 + E22) / 2
        b = E12 * E12 / 4 - E11 * E22
        E1 = a + sqrt(a * a + b)
        E2 = a - sqrt(a * a + b)

        # Jarasjarungkiat et al. 2008 (applicable to isotropic and orthotropic materials)
        # if wrinkling model counter for current element is below iteration goal
        if wrinkling_iter < iter_goal:

            # penalty factor
            if sigma_max <= S2:

                P[el] = 1

            else:

                P[el] = sigma_max / S2

            # slack state
            if E1 <= 0:

                C_mod_11 *= P[el]
                C_mod_12 *= P[el]
                C_mod_21 *= P[el]
                C_mod_22 *= P[el]
                C_mod_33 *= P[el]

                state[el] = 0

            # wrinkling state
            elif S2 <= 0 < E1:
                # TODO: initialize alpha with theta and check if it makes a difference in number of Newton iterations
                # find wrinkling axis
                nIter = 0
                alpha = 0

                # start Newton iterations
                while fabs(f2) > 1E-8:

                    s = sin(alpha)
                    c = cos(alpha)
                    s2 = s * s
                    c2 = c * c
                    sc = s * c

                    f2 = -(S11*s2 - 2*S12*sc + S22*c2) * \
                         (-2*C33*(-s2 + c2)*sc -
                          (C11*s2 + C12*c2)*sc +
                          (C12*s2 + C22*c2)*sc) + \
                         (4*C33*s2*c2 + (C11*s2 + C12*c2) *
                          s2 + (C12*s2 + C22*c2)*c2) \
                    *(-S11*sc + S12*(-s2 + c2) +
                      S22*sc)

                    df2 = (-S11*s2 + 2*S12*sc - S22*c2) * \
                          (2*C33*(-s2 + c2)*s2 -
                           2*C33*(-s2 + c2)*c2 +
                           8*C33*s2*c2 -
                           (-C11*s2 - C12*c2)*s2 +
                           (-C11*s2 - C12*c2)*c2 -
                           (C12*s2 + C22*c2)*s2 +
                           (C12*s2 + C22*c2)*c2 +
                           (-2*C11*sc +
                            2*C12*sc)*sc +
                           (2*C12*sc -
                            2*C22*sc)*sc) + \
                          (4*C33*s2*c2 +
                           (C11*s2 + C12*c2)*s2 +
                           (C12*s2 + C22*c2)*c2)*\
                          (S11*s2 - S11*c2 - 4*S12*sc -
                           S22*s2 + S22*c2) + \
                          (-S11*sc + S12*(-s2 + c2) +
                           S22*sc)*(-8*C33*s2*s*c +
                                                       8*C33*sc**3 +
                                                       2*(C11*s2 + C12*c2)
                                                       *sc -
                                                       2*(C12*s2 + C22*c2)*
                                                       sc +
                                                       (2*C11*sc -
                                                        2*C12*sc)*s2 +
                                                       (2*C12*sc -
                                                        2*C22*sc)*c2) + \
                          (-2*C33*(-s2 + c2)*sc -
                           (C11*s2 + C12*c2)*sc +
                           (C12*s2 + C22*c2)*sc)* \
                          (-2*S11*sc - 2*S12*s2 + 2*S12*c2 +
                           2*S22*sc)

                    if df2 == 0:
                        # print("df2 = 0 in wrinkling Newton")
                        # print("el = {}".format(el))
                        break

                    dalpha = - f2 / df2

                    alpha += dalpha

                    if nIter > maxIter:
                        # if fabs(f2) / 1E-8 > 1E5:
                        #     print('Newton residual in membraneWrinklingJarasjarungkiat not fully converged.')
                        # print(fabs(f2) / 1E-8)
                        break

                    nIter += 1

                # S1_hat_tilde_tilde
                s = sin(alpha)
                c = cos(alpha)
                s2 = s * s
                c2 = c * c
                S1_hat_tilde = 2*C33*(E12 - 2*(-S11*s2 + 2*S12*s*c - S22*c2)*s*c/(4*C33*s2*c2 + (C11*s2 + C12*c2)*s2 + (C12*s2 + C22*c2)*c2))*s*c + (E11 + (-S11*s2 + 2*S12*s*c - S22*c2)*s2/(4*C33*s2*c2 + (C11*s2 + C12*c2)*s2 + (C12*s2 + C22*c2)*c2))*(C11*c2 + C12*s2) + (E22 + (-S11*s2 + 2*S12*s*c - S22*c2)*c2/(4*C33*s2*c2 + (C11*s2 + C12*c2)*s2 + (C12*s2 + C22*c2)*c2))*(C12*c2 + C22*s2)

                # apply phase angle shift if S1_hat_tilde is negative
                if S1_hat_tilde < 0:
                    alpha -= 3.141592653589 / 2

                s = sin(alpha)
                c = cos(alpha)
                s2 = s * s
                c2 = c * c

                # transform back
                C_mod_11 = -2*c*s*(P[el]*s2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) - 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + c2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2))) + c2*(P[el]*s2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) - 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + c2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))) + s2*(-2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) + P[el]*s2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))

                C_mod_12 = 2*c*s*(P[el]*s2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) - 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + c2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2))) + c2*(-2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) + P[el]*s2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + s2*(P[el]*s2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) - 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + c2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))))

                C_mod_13 = c*s*(P[el]*s2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) - 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + c2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))) - c*s*(-2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) + P[el]*s2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + (c2 - s2)*(P[el]*s2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) - 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + c2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)))

                C_mod_21 = -2*c*s*(P[el]*c2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + s2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2))) + c2*(P[el]*c2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + s2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))) + s2*(2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*s2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)))

                C_mod_22 = 2*c*s*(P[el]*c2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + s2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2))) + c2*(2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*s2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))) + s2*(P[el]*c2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + s2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))))

                C_mod_23 = c*s*(P[el]*c2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + s2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))) - c*s*(2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*s2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))) + (c2 - s2)*(P[el]*c2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + s2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)))

                C_mod_31 = -2*c*s*(-P[el]*c*s*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + c*s*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)) + (c2 - s2)*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s))) + c2*(-P[el]*c*s*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + c*s*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + (c2 - s2)*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s))) + s2*(P[el]*c*s*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) - P[el]*c*s*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*(c2 - s2)*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)))

                C_mod_32 = 2*c*s*(-P[el]*c*s*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + c*s*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)) + (c2 - s2)*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s))) + c2*(P[el]*c*s*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) - P[el]*c*s*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*(c2 - s2)*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s))) + s2*(-P[el]*c*s*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + c*s*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + (c2 - s2)*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)))

                C_mod_33 = c*s*(-P[el]*c*s*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + c*s*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + (c2 - s2)*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s))) - c*s*(P[el]*c*s*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) - P[el]*c*s*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*(c2 - s2)*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s))) + (c2 - s2)*(-P[el]*c*s*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + c*s*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)) + (c2 - s2)*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)))

                state[el] = 1

            else:

                # taut state
                state[el] = 2

            alpha_array[el] = alpha

        else:

            # slack state
            if state[el] == 0:

                C_mod_11 *= P[el]
                C_mod_12 *= P[el]
                C_mod_21 *= P[el]
                C_mod_22 *= P[el]
                C_mod_33 *= P[el]

            # wrinkling state
            elif state[el] == 1:

                alpha = alpha_array[el]

                s = sin(alpha)
                c = cos(alpha)
                s2 = s * s
                c2 = c * c

                # transform back
                C_mod_11 = -2*c*s*(P[el]*s2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) - 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + c2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2))) + c2*(P[el]*s2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) - 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + c2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))) + s2*(-2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) + P[el]*s2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))
                C_mod_12 = 2*c*s*(P[el]*s2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) - 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + c2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2))) + c2*(-2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) + P[el]*s2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + s2*(P[el]*s2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) - 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + c2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))))

                C_mod_13 = c*s*(P[el]*s2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) - 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + c2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))) - c*s*(-2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) + P[el]*s2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + (c2 - s2)*(P[el]*s2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) - 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + c2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)))

                C_mod_21 = -2*c*s*(P[el]*c2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + s2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2))) + c2*(P[el]*c2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + s2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))) + s2*(2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*s2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)))

                C_mod_22 = 2*c*s*(P[el]*c2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + s2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2))) + c2*(2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*s2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))) + s2*(P[el]*c2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + s2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))))

                C_mod_23 = c*s*(P[el]*c2*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + 2*c*s*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)) + s2*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)))) - c*s*(2*P[el]*c*s*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)) + P[el]*c2*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*s2*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))) + (c2 - s2)*(P[el]*c2*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + 2*c*s*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)) + s2*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)))

                C_mod_31 = -2*c*s*(-P[el]*c*s*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + c*s*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)) + (c2 - s2)*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s))) + c2*(-P[el]*c*s*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + c*s*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + (c2 - s2)*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s))) + s2*(P[el]*c*s*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) - P[el]*c*s*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*(c2 - s2)*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s)))

                C_mod_32 = 2*c*s*(-P[el]*c*s*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + c*s*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)) + (c2 - s2)*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s))) + c2*(P[el]*c*s*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) - P[el]*c*s*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*(c2 - s2)*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s))) + s2*(-P[el]*c*s*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + c*s*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + (c2 - s2)*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s)))

                C_mod_33 = c*s*(-P[el]*c*s*(-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2)) + c*s*(4*C33*c2*s2 + c2*(C11*c2 + C12*s2) + s2*(C12*c2 + C22*s2) - (-4*C33*c2*s2 + c2*(C11*s2 + C12*c2) + s2*(C12*s2 + C22*c2))*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2))/(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2))) + (c2 - s2)*(2*C33*c*s*(c2 - s2) + c2*(-C11*c*s + C12*c*s) + s2*(-C12*c*s + C22*c*s))) - c*s*(P[el]*c*s*(-4*C33*c2*s2 + c2*(C12*c2 + C22*s2) + s2*(C11*c2 + C12*s2)) - P[el]*c*s*(4*C33*c2*s2 + c2*(C12*s2 + C22*c2) + s2*(C11*s2 + C12*c2)) + P[el]*(c2 - s2)*(-2*C33*c*s*(c2 - s2) + c2*(-C12*c*s + C22*c*s) + s2*(-C11*c*s + C12*c*s))) + (c2 - s2)*(-P[el]*c*s*(-2*C33*c*s*(c2 - s2) - c*s*(C11*s2 + C12*c2) + c*s*(C12*s2 + C22*c2)) + c*s*(2*C33*c*s*(c2 - s2) - c*s*(C11*c2 + C12*s2) + c*s*(C12*c2 + C22*s2)) + (c2 - s2)*(C33*(c2 - s2)**2 - c*s*(-C11*c*s + C12*c*s) + c*s*(-C12*c*s + C22*c*s)))

        SLocal[0] = C_mod_11 * ELocal[0] + C_mod_12 * ELocal[1] + C_mod_13 * ELocal[2]
        SLocal[1] = C_mod_21 * ELocal[0] + C_mod_22 * ELocal[1] + C_mod_23 * ELocal[2]
        SLocal[2] = C_mod_31 * ELocal[0] + C_mod_32 * ELocal[1] + C_mod_33 * ELocal[2]


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int membraneWrinklingState(double [:] ELocal,
                                double [:] SLocal,
                                unsigned int [:] state,
                                unsigned int nelemsCable,
                                Py_ssize_t el) nogil:

    cdef double a, b, E1, E2, S2

    with cython.boundscheck(False):
        
        # determine principal strain
        a = (ELocal[0] + ELocal[1]) / 2
        b = ELocal[2] * ELocal[2] / 4 - ELocal[0] * ELocal[1]
        E1 = a + sqrt(a * a + b)
        E2 = a - sqrt(a * a + b)

        # determine principal stress
        a = (SLocal[0] + SLocal[1]) / 2
        b = SLocal[2] * SLocal[2] - SLocal[0] * SLocal[1]
        S2 = a - sqrt(a * a + b)

        # taut state, treat normally
        if S2 > 0:

            state[el] = 2

        # wrinkling state, S2 < 0 AND E1 > 0
        elif S2 <= 0 < E1:

            # principal strains in tensor form
            state[el] = 1

        # slack state, stress = 0
        else:

            # set stress and to null
            state[el] = 0
