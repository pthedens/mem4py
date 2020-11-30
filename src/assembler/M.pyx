# cython: language_level=3
# cython: boundcheck=False
import numpy as np
cimport numpy as np
cimport cython

from src.elements.cable cimport cable2DFintAndK
from src.elements.cable cimport cable3DFintAndK
from src.elements.membrane cimport membrane2DStrain
from src.elements.membrane cimport membrane2DKmat
from src.elements.membrane cimport membrane3DStrain
from src.elements.membrane cimport membrane3DKmat
from src.elements.membrane cimport membraneStress
from src.elements.membrane cimport membraneWrinklingJarasjarungkiat
from src.helper.sparse cimport addRows


cdef extern from "math.h":
    double sqrt(double m)
    double cos(double m)
    double sin(double m)
    double atan2(double m, double n)
    double fabs(double m)


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int assemble_M_DR(double [:] M,
                       double [:] Minv,
                       double [:] X,
                       double [:] Y,
                       double [:] Z,
                       int [:, ::1] NMem,
                       int [:, ::1] NCable,
                       long double [:] J11Vec,
                       long double [:] J22Vec,
                       long double [:] J12Vec,
                       unsigned int nelemsMem,
                       unsigned int nelemsCable,
                       unsigned int ndof,
                       unsigned int nPressurized,
                       unsigned int nFSI,
                       double [:] E3,
                       double [:] E2,
                       double [:] nu,
                       double [:] t,
                       double [:] area2,
                       double [:] thetaVec,
                       double [:] area3,
                       double [:] L0,
                       double [:] Fint,
                       double [:] p,
                       double [:] pFSI,
                       unsigned int [:] order,
                       unsigned int [:] indptr,
                       unsigned int [:] elPressurised,
                       unsigned int [:] elFSI,
                       unsigned int [:] state,
                       double [:] data,
                       double [:] diagK,
                       double * alpha,
                       double [:] sumRowWithDiag,
                       unsigned int dim,
                       double * IE,
                       str method,
                       double lam,
                       double [:] P,
                       double [:] alpha_array,
                       unsigned int wrinkling_iter,
                       unsigned int iter_goal,
                       double sigma_max,
                       double loadStep,
                       double [:] ELocal,
                       double [:] SLocal) except -1:
    """
    Mass matrix (identity) assembly according to Alamatian et al. 2012,
    largest eigenvalue of K scales mass for optimal DR convergence

    :param M:           mass matrix in vector form
    :param Minv:        inverted mass matrix in vector form
    :param X:           X-coordinates (current configuration)
    :param Y:           Y-coordinates (current configuration)
    :param Z:           Z-coordinates (current configuration)
    :param N:           element connectivity matrix
    :param J11Vec:      transformation parameters between initial and current configuration
    :param J22Vec:      transformation parameters between initial and current configuration
    :param J12Vec:      transformation parameters between initial and current configuration
    :param nelems:      number of elements
    :param nnodes:      number of nodes
    :param E:           Young's modulus
    :param poisson:     Poisson's ratio
    :param t:           material thickness
    :param thetaVec:    angle between local coordinate system and fibre direction
    :param area3[el]:     element area

    :return:            void, M and Minv are changed in memory view
    """
    cdef:

        double S1, S2, SE

        Py_ssize_t el, i, index = 0, indP = 0, indFSI = 0
        
        double theta, pLocal
        unsigned int wrinklingFlag

    Fint[...] = 0
    data[...] = 0
    diagK[...] = 0
    IE[0] = 0

    if dim == 2:

        # Loop over all cable elements
        for el in range(nelemsCable):

            # if 0 -> bar element, if 1 -> cable element
            wrinklingFlag = NCable[el, 3]

            cable2DFintAndK(X, Y, NCable, L0[el], el, area2[el], E2[el], index,
                            Fint, data, diagK, order, &SE, wrinklingFlag)

            # increase index by length of entries in local stiffness matrix Kloc (6 x 6)
            index += 16

            # strain energy
            IE[0] += SE

        # Loop over all membrane elements
        for el in range(nelemsMem):

            pLocal = 0

            # determine strain
            membrane2DStrain(X, Y, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], ELocal)

            # determine stressVoigt, S1, S2, theta
            membraneStress(ELocal, SLocal, &S1, &S2, &theta, E3[el], nu[el])

            if NMem[el, 4] == 1 and wrinkling_iter != 0:
                membraneWrinklingJarasjarungkiat(ELocal, SLocal, state, S2, theta, el, E3[el], nu[el],
                                                 P, alpha_array, wrinkling_iter, iter_goal, sigma_max)

            # determine element pressure for KMem
            if nPressurized != 0:
                if el == elPressurised[indP]:
                    pLocal = p[indP]
                    indP += 1
            if nFSI != 0:
                if el == elFSI[indFSI]:
                    pLocal += pFSI[indFSI]
                    indFSI += 1

            membrane2DKmat(X, Y, NMem, SLocal, Fint, t[el], area3[el], pLocal*loadStep, el, J11Vec[el], J22Vec[el],
                           J12Vec[el], E3[el], nu[el], data, diagK, index, order)

            # increase index by length of entries in local stiffness matrix Kloc (9 x 9)
            index += 36

            # strain energy
            IE[0] += area3[el] * t[el] * (ELocal[0] * SLocal[0] + ELocal[1] * SLocal[1] + 2 * ELocal[2] * SLocal[2])

    elif dim == 3:

        # Loop over all cable elements
        for el in range(nelemsCable):

            # if 0 -> bar element, if 1 -> cable element
            wrinklingFlag = NCable[el, 3]

            cable3DFintAndK(X, Y, Z, NCable, L0[el], el, area2[el], E2[el], index, Fint,
                            data, diagK, order, &SE, wrinklingFlag)

            # increase index by length of entries in local stiffness matrix Kloc (6 x 6)
            index += 36

            # strain energy
            IE[0] += SE

        # Loop over all membrane elements
        for el in range(nelemsMem):

            pLocal = 0

            # determine strain
            membrane3DStrain(X, Y, Z, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], ELocal)

            # determine stressVoigt, S1, S2, theta
            membraneStress(ELocal, SLocal, &S1, &S2, &theta, E3[el], nu[el])

            if NMem[el, 4] == 1 and wrinkling_iter != 0:
                membraneWrinklingJarasjarungkiat(ELocal, SLocal, state, S2, theta, el, E3[el], nu[el],
                                                 P, alpha_array, wrinkling_iter, iter_goal, sigma_max)

            # determine element pressure for KMem
            if nPressurized != 0:
                if el == elPressurised[indP]:
                    pLocal = p[indP]
                    indP += 1
            if nFSI != 0:
                if el == elFSI[indFSI]:
                    pLocal += pFSI[indFSI]
                    indFSI += 1

            membrane3DKmat(X, Y, Z, NMem, SLocal, Fint, t[el], area3[el], pLocal*loadStep, el, J11Vec[el], J22Vec[el],
                           J12Vec[el], E3[el], nu[el], data, diagK, index, order)

            # increase index by length of entries in local stiffness matrix Kloc (9 x 9)
            index += 81

            # strain energy
            IE[0] += area3[el] * t[el] * (ELocal[0] * SLocal[0] + ELocal[1] * SLocal[1] + 2 * ELocal[2] * SLocal[2])
    
    addRows(data, indptr, ndof, diagK, alpha, M, Minv, sumRowWithDiag, wrinklingFlag, dim, method, lam)

    IE[0] *= 0.5


cdef int assembleM(int [:, ::1] NMem,
                   int [:, ::1] NCable,
                   double [:] M,
                   double [:] Minv,
                   double [:] area3,
                   double [:] L0,
                   double [:] t,
                   double [:] area2,
                   double [:] rho3,
                   double [:] rho2,
                   unsigned int nelemsMem,
                   unsigned int nelemsCable,
                   unsigned int dim) except -1:


    cdef:

        Py_ssize_t el
        Py_ssize_t dof_1, dof_2, dof_3, dof_4, dof_5, dof_6, dof_7, dof_8, dof_9, 

        double MLocal

    M[...] = 0

    if dim == 2:

        for el in range(nelemsCable):

            dof_1 = 2 * (NCable[el, 1] + 1) - 2
            dof_2 = 2 * (NCable[el, 1] + 1) - 1
            dof_3 = 2 * (NCable[el, 2] + 1) - 2
            dof_4 = 2 * (NCable[el, 2] + 1) - 1

            MLocal = rho2[el] * L0[el] * area2[el] / 2

            M[dof_1] += MLocal
            M[dof_2] += MLocal
            M[dof_3] += MLocal
            M[dof_4] += MLocal

            Minv[dof_1] += 1 / MLocal
            Minv[dof_2] += 1 / MLocal
            Minv[dof_3] += 1 / MLocal
            Minv[dof_4] += 1 / MLocal

        for el in range(nelemsMem):

            dof_1 = 2 * (NMem[el, 1] + 1) - 2
            dof_2 = 2 * (NMem[el, 1] + 1) - 1
            dof_3 = 2 * (NMem[el, 2] + 1) - 2
            dof_4 = 2 * (NMem[el, 2] + 1) - 1
            dof_5 = 2 * (NMem[el, 3] + 1) - 2
            dof_6 = 2 * (NMem[el, 3] + 1) - 1

            MLocal = rho3[el] * area3[el] * t[el] / 3

            M[dof_1] += MLocal
            M[dof_2] += MLocal
            M[dof_3] += MLocal
            M[dof_4] += MLocal
            M[dof_5] += MLocal
            M[dof_6] += MLocal

            Minv[dof_1] += 1 / MLocal
            Minv[dof_2] += 1 / MLocal
            Minv[dof_3] += 1 / MLocal
            Minv[dof_4] += 1 / MLocal
            Minv[dof_5] += 1 / MLocal
            Minv[dof_6] += 1 / MLocal

    elif dim == 3:

        for el in range(nelemsCable):

            dof_1 = 3 * (NCable[el, 1] + 1) - 3
            dof_2 = 3 * (NCable[el, 1] + 1) - 2
            dof_3 = 3 * (NCable[el, 1] + 1) - 1
            dof_4 = 3 * (NCable[el, 2] + 1) - 3
            dof_5 = 3 * (NCable[el, 2] + 1) - 2
            dof_6 = 3 * (NCable[el, 2] + 1) - 1

            MLocal = rho2[el] * L0[el] * area2[el] / 2

            M[dof_1] += MLocal
            M[dof_2] += MLocal
            M[dof_3] += MLocal
            M[dof_4] += MLocal
            M[dof_5] += MLocal
            M[dof_6] += MLocal

            Minv[dof_1] += 1 / MLocal
            Minv[dof_2] += 1 / MLocal
            Minv[dof_3] += 1 / MLocal
            Minv[dof_4] += 1 / MLocal
            Minv[dof_5] += 1 / MLocal
            Minv[dof_6] += 1 / MLocal

        for el in range(nelemsMem):

            dof_1 = 3 * (NMem[el, 1] + 1) - 3
            dof_2 = 3 * (NMem[el, 1] + 1) - 2
            dof_3 = 3 * (NMem[el, 1] + 1) - 1
            dof_4 = 3 * (NMem[el, 2] + 1) - 3
            dof_5 = 3 * (NMem[el, 2] + 1) - 2
            dof_6 = 3 * (NMem[el, 2] + 1) - 1
            dof_7 = 3 * (NMem[el, 3] + 1) - 3
            dof_8 = 3 * (NMem[el, 3] + 1) - 2
            dof_9 = 3 * (NMem[el, 3] + 1) - 1

            MLocal = rho3[el] * area3[el] * t[el] / 3

            M[dof_1] += MLocal
            M[dof_2] += MLocal
            M[dof_3] += MLocal
            M[dof_4] += MLocal
            M[dof_5] += MLocal
            M[dof_6] += MLocal
            M[dof_7] += MLocal
            M[dof_8] += MLocal
            M[dof_9] += MLocal

            Minv[dof_1] += 1 / MLocal
            Minv[dof_2] += 1 / MLocal
            Minv[dof_3] += 1 / MLocal
            Minv[dof_4] += 1 / MLocal
            Minv[dof_5] += 1 / MLocal
            Minv[dof_6] += 1 / MLocal
            Minv[dof_7] += 1 / MLocal
            Minv[dof_8] += 1 / MLocal
            Minv[dof_9] += 1 / MLocal
