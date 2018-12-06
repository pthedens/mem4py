cdef void bar3DStrainGreen(double l,
                           double L0,
                           double * strainBar)


cdef void bar3DCauchyStress(double [:] X,
                            double [:] Y,
                            double [:] Z,
                            int [:, ::1] N,
                            double L0,
                            double EBar,
                            double * strainBar,
                            double * stressBar,
                            unsigned int el)


cdef void bar3DFintAndK(double [:] X,
                        double [:] Y,
                        double [:] Z,
                        int [:, ::1] N,
                        double L0,
                        unsigned int el,
                        double areaBar,
                        double E,
                        double [:] Fint,
                        unsigned int [:] allDofBar,
                        double [:, ::1] Klocal)