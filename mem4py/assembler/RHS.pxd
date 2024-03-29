# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef int assembleRHS(double [:] X,
                     double [:] Y,
                     double [:] Z,
                     double [:] u,
                     int [:, ::1] NMem,
                     int [:, ::1] NCable,
                     double [:] p,
                     double [:] RHS,
                     double [:] RHS0,
                     unsigned int [:] elPressurised,
                     unsigned int [:] elFSI,
                     double [:] area3,
                     double [:] L0,
                     int gravity,
                     unsigned int nelemsMem,
                     unsigned int nelemsCable,
                     unsigned int nPressurised,
                     unsigned int nFSI,
                     double [:] t,
                     double [:] rho3,
                     double [:] area2,
                     double [:] rho2,
                     double [:] g,
                     double [:] Sx,
                     double [:] Sy,
                     double [:] Sz,
                     double [:] pFSI,
                     double [:, ::1] loadedBCNodes,
                     double [:, ::1] loadedBCEdges,
                     double [:, ::1] loadedBCSurface,
                     unsigned int RHS0flag,
                     double loadStep,
                     unsigned int dim,
                     double [:] force_vector,
                     double [:] E2,
                     double [:] pre_stress_cable,
                     double [:] pre_strain_cable,
                     double [:, ::1] pre_stress_membrane,
                     double [:, ::1] pre_strain_membrane,
                     unsigned int [:] pre_active,
                     long double [:] J11Vec,
                     long double [:] J22Vec,
                     long double [:] J12Vec,
                     double [:] thetaVec,
                     double [:] E3,
                     double [:] nu,
                     object aero) except -1


cdef int pre_membrane_2d(double [:] RHS,
                         int [:, ::1] NMem,
                         double [:] X,
                         double [:] Y,
                         double [:] u,
                         long double [:] J11Vec,
                         long double [:] J22Vec,
                         long double [:] J12Vec,
                         double [:] areaVec,
                         double [:] thetaVec,
                         double [:] nu,
                         double [:] E3,
                         double [:] t,
                         double [:, ::1] pre_stress_membrane,
                         double [:, ::1] pre_strain_membrane,
                         unsigned int nelemenMem) except -1