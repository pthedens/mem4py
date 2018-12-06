import numpy as np
cimport numpy as np
cimport cython

from src.elements.bar2D cimport bar2DFintAndK
from src.elements.membrane2D cimport membrane2DStrain
from src.elements.membrane2D cimport membrane2DStress
from src.elements.membrane2D cimport membrane2DBmat
from src.elements.membrane2D cimport membrane2DKmat

from src.ceygen.math cimport dot_mv

from src.materialModels.StVenantIso cimport CmatStVenantIsotropic

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int assembleK2D(int [:, ::1] NMem,
                     int [:, ::1] NBar,
                     double [:] X,
                     double [:] Y,
                     double [:] J11Vec,
                     double [:] J22Vec,
                     double [:] J12Vec,
                     double [:] areaVec,
                     double [:] L0,
                     double [:] Fint,
                     double [:] data,
                     unsigned int [:] row,
                     unsigned int [:] col,
                     double EMem,
                     double nu,
                     double t,
                     double EBar,
                     double areaBar,
                     unsigned int nelemsMem,
                     unsigned int nelemsBar,
                     unsigned int iteration) except -1:

    # type def local variables
    cdef:

        # bar elements
        unsigned int [:] allDofBar = np.zeros(4, dtype=np.uintc)

        double [:] FintBar = np.zeros(4, dtype=np.double)
        double [:, ::1] KBar = np.zeros((4, 4), dtype=np.double)

        # membrane elements
        unsigned int [:] allDofMem = np.zeros(6, dtype=np.uintc)

        double [:] ELocal =  np.zeros(3, dtype=np.double)
        double [:] SLocal =  np.zeros(3, dtype=np.double)
        double [:] s =       np.zeros(3, dtype=np.double)
        double [:] FintMem = np.zeros(6, dtype=np.double)

        double [:, :] BmatLocal = np.zeros((3, 6), dtype=np.double)
        double [:, :] KMem =      np.zeros((6, 6), dtype=np.double)
        double [:, :] Cmat =      np.zeros((3, 3), dtype=np.double)

        unsigned int el, i, j, index = 0, runner, ii
        double a, b, S1, S2, E1, E2

        double theta

    Fint[...] = 0

    # loop through bar elements
    for el in range(nelemsBar):

        bar2DFintAndK(X, Y, NBar, L0[el], el, areaBar, EBar, Fint, allDofBar, KBar)

        # row major flattening
        runner = 0
        for i in range(4):

            # add local to global Fint
            Fint[allDofBar[i]] += FintBar[i]

            for j in range(4):
                data[index + runner] = KBar[i, j]

                runner += 1

        # increase index by length of entries in local stiffness matrix Kloc (4 x 4)
        index += 16

    # constitutive matrix
    CmatStVenantIsotropic(EMem, nu, Cmat)

    if iteration == 0:

        # Loop over all membrane elements
        for el in range(nelemsMem):

            # determine strain
            membrane2DStrain(X, Y, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], allDofMem, ELocal)

            # determine stressVoigt, S1, S2, theta
            membrane2DStress(Cmat, ELocal, SLocal, &S1, &S2, &theta)

            # determine Bmat and local stress s
            membrane2DBmat(X, Y, NMem, J11Vec[el], J22Vec[el], J12Vec[el], el, SLocal, s, BmatLocal)

            # determine KMem
            membrane2DKmat(BmatLocal, Cmat, KMem, s, t, areaVec[el])

            # local internal force vector
            dot_mv(BmatLocal.T, SLocal, FintMem)

            # row major flattening
            runner = 0
            for i in range(6):

                # add local to global Fint
                Fint[allDofMem[i]] += FintMem[i] * areaVec[el] * t

                for j in range(6):
                    row[index + runner] = allDofMem[i]
                    col[index + runner] = allDofMem[j]
                    data[index + runner] = KMem[i, j]
                    runner += 1

            # increase index by length of entries in local stiffness matrix Kloc (6 x 6)
            index += 36

    else:

        # Loop over all membrane elements
        for el in range(nelemsMem):

            # determine strain
            membrane2DStrain(X, Y, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], allDofMem, ELocal)

            # determine stressVoigt, S1, S2, theta
            membrane2DStress(Cmat, ELocal, SLocal, &S1, &S2, &theta)

            # determine Bmat and local stress s
            membrane2DBmat(X, Y, NMem, J11Vec[el], J22Vec[el], J12Vec[el], el, SLocal, s, BmatLocal)

            # determine KMem
            membrane2DKmat(BmatLocal, Cmat, KMem, s, t, areaVec[el])

            # local internal force vector
            dot_mv(BmatLocal.T, SLocal, FintMem)

            # row major flattening
            runner = 0
            for i in range(6):

                # add local to global Fint
                Fint[allDofMem[i]] += FintMem[i] * areaVec[el] * t

                for j in range(6):
                    data[index + runner] = KMem[i, j]
                    runner += 1

            # increase index by length of entries in local stiffness matrix Kloc (6 x 6)
            index += 36
