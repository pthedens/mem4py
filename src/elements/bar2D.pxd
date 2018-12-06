cdef void bar2DStrainGreen(double l,
                           double L0,
                           double * strainBar)


cdef void bar2DCauchyStress(double [:] X,
                            double [:] Y,
                            int [:, ::1] N,
                            double L0,
                            double EBar,
                            double * strainBar,
                            double * stressBar,
                            unsigned int el)


cdef void bar2DFintAndK(double [:] X,
                        double [:] Y,
                        int [:, ::1] N,
                        double L0,
                        unsigned int el,
                        double areaBar,
                        double E,
                        double [:] Fint,
                        unsigned int [:] allDofBar,
                        double [:, ::1] Klocal)