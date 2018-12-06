cdef void membrane3DStrain(double [:] X,
                           double [:] Y,
                           double [:] Z,
                           int [:, ::1] N,
                           unsigned int el,
                           double J11,
                           double J22,
                           double J12,
                           unsigned int [:] allDofMem,
                           double [:] ELocal)


cdef void membrane3DStress(double [:, :] Cmat,
                           double [:] strainVoigt,
                           double [:] stressVoigt,
                           double * S1,
                           double * S2,
                           double * theta)


cdef void membrane3DBmat(double [:] X,
                         double [:] Y,
                         double [:] Z,
                         int [:, ::1] N,
                         double J11,
                         double J22,
                         double J12,
                         unsigned int el,
                         double [:] SLocal,
                         double [:] s,
                         double [:, :] BmatLocal)


cdef void membrane3DKmat(double [:] X,
                         double [:] Y,
                         double [:] Z,
                         int [:, ::1] N,
                         double [:, :] BmatLocal,
                         double [:, :] Cmat,
                         double [:, :] KMem,
                         double [:] s,
                         double t,
                         double area,
                         double p,
                         unsigned int el)
