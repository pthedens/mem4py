import numpy as np
cimport numpy as np
cimport cython

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import norm, eigs

from src.ceygen.math cimport dot_mm, dot_mv

from src.materialModels.StVenantIso cimport CmatStVenantIsotropic

from src.elements.bar3D cimport bar3DFintAndK

from src.elements.membrane3D cimport membrane3DStrain
from src.elements.membrane3D cimport membrane3DStress
from src.elements.membrane3D cimport membrane3DBmat
from src.elements.membrane3D cimport membrane3DKmat

cdef extern from "math.h":
    double sqrt(double m)
    double cos(double m)
    double sin(double m)
    double atan2(double m, double n)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void M3DIsotropic(double [:] M,
                       double [:] Minv,
                       double [:] X,
                       double [:] Y,
                       double [:] Z,
                       int [:, ::1] N,
                       double [:] J11Vec,
                       double [:] J22Vec,
                       double [:] J12Vec,
                       unsigned int nelems,
                       unsigned int nnodes,
                       double E,
                       double poisson,
                       double t,
                       double [:] thetaVec,
                       double [:] areaVec,
                       double [:] Fint):

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
    :param areaVec:     element area

    :return:            void, M and Minv are changed in memory view
    """
    cdef unsigned int [:] allDofEMem = np.zeros(9, dtype=np.uintc)
    cdef unsigned int [:] row = np.empty(nelems * 81, dtype=np.uintc)
    cdef unsigned int [:] col = np.empty(nelems * 81, dtype=np.uintc)
    cdef double [:] data = np.empty(nelems * 81, dtype=np.double)
    cdef double [:] ELocal = np.zeros(3, dtype=np.double)
    cdef double [:] SLocal = np.zeros(3, dtype=np.double)
    cdef double [:] s = np.zeros(3, dtype=np.double)
    cdef double [:] FintElement = np.zeros(9, dtype=np.double)

    cdef double [:, :] BmatCurv = np.zeros((3, 9), dtype=np.double)
    cdef double [:, :] BmatLocal = np.zeros((3, 9), dtype=np.double)
    cdef double [:, :] Kloc = np.zeros((9, 9), dtype=np.double)
    cdef double [:, :] dBdS = np.zeros((9, 9), dtype=np.double)
    cdef double [:, :] Cmat = np.zeros((3, 3), dtype=np.double)
    cdef double [:, :] Q = np.zeros((3, 3), dtype=np.double)

    cdef unsigned int el, i, j, index = 0, runner, ndof = 3 * nnodes
    cdef double g11, g12, g22

    Fint[...] = 0

    # Loop over all elements
    for el in range(nelems):

        # Find degrees of freedom from current element
        allDofEMem[0] = 3 * (N[el, 1] + 1) - 3
        allDofEMem[1] = 3 * (N[el, 1] + 1) - 2
        allDofEMem[2] = 3 * (N[el, 1] + 1) - 1
        allDofEMem[3] = 3 * (N[el, 2] + 1) - 3
        allDofEMem[4] = 3 * (N[el, 2] + 1) - 2
        allDofEMem[5] = 3 * (N[el, 2] + 1) - 1
        allDofEMem[6] = 3 * (N[el, 3] + 1) - 3
        allDofEMem[7] = 3 * (N[el, 3] + 1) - 2
        allDofEMem[8] = 3 * (N[el, 3] + 1) - 1

        # covariant components of the metric tensor in current configuration
        g11 = (X[N[el, 2]] - X[N[el, 1]]) * (X[N[el, 2]] - X[N[el, 1]]) + \
              (Y[N[el, 2]] - Y[N[el, 1]]) * (Y[N[el, 2]] - Y[N[el, 1]]) + \
              (Z[N[el, 2]] - Z[N[el, 1]]) * (Z[N[el, 2]] - Z[N[el, 1]])

        g12 = (X[N[el, 2]] - X[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]]) + \
              (Y[N[el, 2]] - Y[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]]) + \
              (Z[N[el, 2]] - Z[N[el, 1]]) * (Z[N[el, 3]] - Z[N[el, 1]])

        g22 = (X[N[el, 3]] - X[N[el, 1]]) * (X[N[el, 3]] - X[N[el, 1]]) + \
              (Y[N[el, 3]] - Y[N[el, 1]]) * (Y[N[el, 3]] - Y[N[el, 1]]) + \
              (Z[N[el, 3]] - Z[N[el, 1]]) * (Z[N[el, 3]] - Z[N[el, 1]])

        # local strain (Cartesian coordinates), ELocal = Q * ECurv, ECurv = 0.5 * (g_ab - G_ab)
        ELocal[0] = 0.5 * g11 / (J11Vec[el] * J11Vec[el]) - 0.5
        ELocal[1] = 0.5 * (g22 * J11Vec[el] -
                           g12 * J12Vec[el] +
                           g11 * J12Vec[el] * J12Vec[el] / J11Vec[el] -
                           g12 * J12Vec[el]) / (J11Vec[el] * J22Vec[el] * J22Vec[el]) - \
                    0.5
        ELocal[2] = (g12 - g11 * J12Vec[el] / J11Vec[el]) / (J11Vec[el] * J22Vec[el]) / 2

        # constitutive matrix
        CmatStVenantIsotropic(E, poisson, Cmat)

        # determine stress in fibre coordinate system
        dot_mv(Cmat, ELocal, SLocal)

        # fill Q matrix
        Q[0, 0] = 1 / (J11Vec[el] * J11Vec[el])
        Q[1, 0] = (J12Vec[el] * J12Vec[el]) / (J11Vec[el] * J11Vec[el] * J22Vec[el] * J22Vec[el])
        Q[1, 1] = 1 / (J22Vec[el] * J22Vec[el])
        Q[1, 2] = - J12Vec[el] / (J11Vec[el] * J22Vec[el] * J22Vec[el])
        Q[2, 0] = - 2 * J12Vec[el] / (J11Vec[el] * J11Vec[el] * J22Vec[el])
        Q[2, 2] = 1 / (J11Vec[el] * J22Vec[el])

        # strain displacement matrix BmatCurv
        BmatCurv[0, 0] = - (X[N[el, 2]] - X[N[el, 1]])
        BmatCurv[0, 1] = - (Y[N[el, 2]] - Y[N[el, 1]])
        BmatCurv[0, 2] = - (Z[N[el, 2]] - Z[N[el, 1]])

        BmatCurv[0, 3] = X[N[el, 2]] - X[N[el, 1]]
        BmatCurv[0, 4] = Y[N[el, 2]] - Y[N[el, 1]]
        BmatCurv[0, 5] = Z[N[el, 2]] - Z[N[el, 1]]

        BmatCurv[1, 0] = - (X[N[el, 3]] - X[N[el, 1]])
        BmatCurv[1, 1] = - (Y[N[el, 3]] - Y[N[el, 1]])
        BmatCurv[1, 2] = - (Z[N[el, 3]] - Z[N[el, 1]])

        BmatCurv[1, 6] = X[N[el, 3]] - X[N[el, 1]]
        BmatCurv[1, 7] = Y[N[el, 3]] - Y[N[el, 1]]
        BmatCurv[1, 8] = Z[N[el, 3]] - Z[N[el, 1]]

        BmatCurv[2, 0] = 2 * X[N[el, 1]] - X[N[el, 2]] - X[N[el, 3]]
        BmatCurv[2, 1] = 2 * Y[N[el, 1]] - Y[N[el, 2]] - Y[N[el, 3]]
        BmatCurv[2, 2] = 2 * Z[N[el, 1]] - Z[N[el, 2]] - Z[N[el, 3]]

        BmatCurv[2, 3] = X[N[el, 3]] - X[N[el, 1]]
        BmatCurv[2, 4] = Y[N[el, 3]] - Y[N[el, 1]]
        BmatCurv[2, 5] = Z[N[el, 3]] - Z[N[el, 1]]

        BmatCurv[2, 6] = X[N[el, 2]] - X[N[el, 1]]
        BmatCurv[2, 7] = Y[N[el, 2]] - Y[N[el, 1]]
        BmatCurv[2, 8] = Z[N[el, 2]] - Z[N[el, 1]]

        # BLocal = Q * BmatCurv
        dot_mm(Q, BmatCurv, BmatLocal)

        # Kloc = B.T * C * B
        dot_mm(BmatLocal.T, dot_mm(Cmat, BmatLocal, BmatCurv), Kloc)

        # Set BmatCurv entries to zeros
        BmatCurv[...] = 0

        # strain stiffness contribution
        dot_mv(Q.T, SLocal, s)

        dBdS[0, 0] = s[0] + s[1] + 2 * s[2]
        dBdS[1, 1] = s[0] + s[1] + 2 * s[2]
        dBdS[2, 2] = s[0] + s[1] + 2 * s[2]

        dBdS[3, 0] = - s[0] - s[2]
        dBdS[4, 1] = - s[0] - s[2]
        dBdS[5, 2] = - s[0] - s[2]

        dBdS[0, 3] = - s[0] - s[2]
        dBdS[1, 4] = - s[0] - s[2]
        dBdS[2, 5] = - s[0] - s[2]

        dBdS[6, 0] = - s[1] - s[2]
        dBdS[7, 1] = - s[1] - s[2]
        dBdS[8, 2] = - s[1] - s[2]

        dBdS[0, 6] = - s[1] - s[2]
        dBdS[1, 7] = - s[1] - s[2]
        dBdS[2, 8] = - s[1] - s[2]

        dBdS[3, 3] = s[0]
        dBdS[4, 4] = s[0]
        dBdS[5, 5] = s[0]

        dBdS[3, 6] = s[2]
        dBdS[4, 7] = s[2]
        dBdS[5, 8] = s[2]

        dBdS[6, 3] = s[2]
        dBdS[7, 4] = s[2]
        dBdS[8, 5] = s[2]

        dBdS[6, 6] = s[1]
        dBdS[7, 7] = s[1]
        dBdS[8, 8] = s[1]

        # local internal force vector
        dot_mv(BmatLocal.T, SLocal, FintElement)

        # row major flattening
        runner = 0
        for i in range(0, 9):

            # add local to global RHS vector
            Fint[allDofEMem[i]] += FintElement[i] * areaVec[el] * t

            for j in range(0, 9):
                row[index + runner] = allDofEMem[i]
                col[index + runner] = allDofEMem[j]
                data[index + runner] = (Kloc[i, j] + dBdS[i, j]) * areaVec[el] * t
                runner += 1

        # increase index by length of entries in local stiffness matrix Kloc (9 x 9)
        index += 81

    # Assemble sparse matrix K2D
    K = coo_matrix((data, (row, col)), shape=(ndof, ndof)).tocsr()
    K.eliminate_zeros()

    cdef double [:] sumColWithDiag = np.empty(ndof, dtype=np.double)
    cdef double [:] sumColNoDiag = np.empty(ndof, dtype=np.double)
    cdef double [:] diagK = np.empty(ndof, dtype=np.double)
    cdef double [:] alphaVec = np.zeros(ndof, dtype=np.double)
    cdef double beta, alphaSqrt, rowSum
    cdef unsigned int ind1, ind2, ind3

    # Diagonal components TODO: CHECK IF ROW OR COL IS FASTER IN CSC FORMAT
    diagK = K.diagonal()

    # Summ columns without diagonal component
    sumColNoDiag = np.asarray(np.abs(K).sum(axis=0)).reshape((ndof,))
    sumColWithDiag = np.asarray(sumColNoDiag) - np.abs(diagK)

    # loop through columns of K and determine alpha
    for i in range(ndof):
        if np.abs(diagK[i]) > 1E-18:
            beta = sumColNoDiag[i] / diagK[i]

            # check for row condition
            if diagK[i] > sumColNoDiag[i]:
                alphaVec[i] = (1 - sqrt(1 - beta * beta)) / beta
            elif 0.5 * sumColNoDiag[i] <= diagK[i] < sumColNoDiag[i]:
                alphaVec[i] = beta - sqrt(beta * beta - 1)
            elif diagK[i] <= 0.5 * sumColNoDiag[i] and np.abs(diagK[i] - sumColNoDiag[i]) < 1E-14:
                alphaVec[i] = (beta + 2 - 2 * sqrt(1 + beta)) / beta

    # find max alpha (sqrt(alpha) actually)
    alphaSqrt = np.max(alphaVec)

    # assemble mass matrix
    for i in range(nnodes):

        ind1 = 3 * (i + 1) - 3
        ind2 = 3 * (i + 1) - 2
        ind3 = 3 * (i + 1) - 1

        M[ind1] = (1 + alphaSqrt * alphaSqrt) / (2 * (1 + alphaSqrt) ** 2) * (sumColWithDiag[ind1] +
                                                                              np.abs(diagK[ind1]))
        M[ind2] = (1 + alphaSqrt * alphaSqrt) / (2 * (1 + alphaSqrt) ** 2) * (sumColWithDiag[ind2] +
                                                                              np.abs(diagK[ind2]))
        M[ind3] = (1 + alphaSqrt * alphaSqrt) / (2 * (1 + alphaSqrt) ** 2) * (sumColWithDiag[ind3] +
                                                                              np.abs(diagK[ind3]))

        Minv[ind1] = 1. / M[ind1]
        Minv[ind2] = 1. / M[ind2]
        Minv[ind3] = 1. / M[ind3]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int M3DExplicit(int [:, ::1] NMem,
                     int [:, ::1] NBar,
                     double [:] Minv,
                     double [:] areaVec,
                     double [:] L0,
                     double t,
                     double areaBar,
                     double rhoMem,
                     double rhoBar,
                     unsigned int nelemsMem,
                     unsigned int nelemsBar) except -1:


    cdef unsigned int [:] allDofMem = np.empty(9, dtype=np.uintc)
    cdef unsigned int [:] allDofBar = np.empty(6, dtype=np.uintc)
    cdef unsigned int el, dof
    cdef double MLocal

    for el in range(nelemsBar):

        allDofBar[0] = 3 * (NBar[el, 1] + 1) - 3
        allDofBar[1] = 3 * (NBar[el, 1] + 1) - 2
        allDofBar[2] = 3 * (NBar[el, 1] + 1) - 1
        allDofBar[3] = 3 * (NBar[el, 2] + 1) - 3
        allDofBar[4] = 3 * (NBar[el, 2] + 1) - 2
        allDofBar[5] = 3 * (NBar[el, 2] + 1) - 1

        MLocal = rhoBar * L0[el] * areaBar

        for dof in range(6):

            Minv[allDofBar[dof]] += 2 / MLocal

    for el in range(nelemsMem):

        allDofMem[0] = 3 * (NMem[el, 1] + 1) - 3
        allDofMem[1] = 3 * (NMem[el, 1] + 1) - 2
        allDofMem[2] = 3 * (NMem[el, 1] + 1) - 1
        allDofMem[3] = 3 * (NMem[el, 2] + 1) - 3
        allDofMem[4] = 3 * (NMem[el, 2] + 1) - 2
        allDofMem[5] = 3 * (NMem[el, 2] + 1) - 1
        allDofMem[6] = 3 * (NMem[el, 3] + 1) - 3
        allDofMem[7] = 3 * (NMem[el, 3] + 1) - 2
        allDofMem[8] = 3 * (NMem[el, 3] + 1) - 1

        MLocal = rhoMem * areaVec[el] * t

        for dof in range(9):

            Minv[allDofMem[dof]] += 3 / MLocal


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef object K3DIsotropic(double [:] X,
                         double [:] Y,
                         double [:] Z,
                         int [:, ::1] NMem,
                         int [:, ::1] NBar,
                         double [:] J11Vec,
                         double [:] J22Vec,
                         double [:] J12Vec,
                         unsigned int nelemsMem,
                         unsigned int nelemsBar,
                         unsigned int ndof,
                         double EMem,
                         double nu,
                         double t,
                         double [:] areaVec,
                         double EBar,
                         double areaBar,
                         double [:] L0,
                         double [:] Fint,
                         double [:] p):

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
    :param areaVec:     element area
    
    :return:            void, M and Minv are changed in memory view
    """

    cdef:

        # bar elements
        unsigned int [:] allDofBar = np.zeros(6, dtype=np.uintc)

        double [:] FintBar = np.zeros(6, dtype=np.double)
        double [:, ::1] KBar = np.zeros((6, 6), dtype=np.double)

        # membrane elements
        unsigned int [:] allDofMem = np.zeros(9, dtype=np.uintc)

        double [:] ELocal =  np.zeros(3, dtype=np.double)
        double [:] SLocal =  np.zeros(3, dtype=np.double)
        double [:] s =       np.zeros(3, dtype=np.double)
        double [:] FintMem = np.zeros(9, dtype=np.double)

        double [:, :] BmatLocal = np.zeros((3, 9), dtype=np.double)
        double [:, :] KMem =      np.zeros((9, 9), dtype=np.double)
        double [:, :] Cmat =      np.zeros((3, 3), dtype=np.double)

        double [:] data =  np.empty(nelemsMem * 81 + nelemsBar * 36, dtype=np.double)
        unsigned int [:] row =  np.empty(nelemsMem * 81 + nelemsBar * 36, dtype=np.uintc)
        unsigned int [:] col =  np.empty(nelemsMem * 81 + nelemsBar * 36, dtype=np.uintc)

        unsigned int el, i, j, index = 0, runner
        double a, b, S1, S2, E1, E2

        double theta

    Fint[...] = 0

    # Loop over all bar elements
    for el in range(nelemsBar):

        bar3DFintAndK(X, Y, Z, NBar, L0[el], el, areaBar, EBar, Fint, allDofBar, KBar)

        # row major flattening
        runner = 0
        for i in range(6):

            # add local to global RHS
            Fint[allDofBar[i]] += FintBar[i]

            for j in range(6):
                data[index + runner] = KBar[i, j]

                runner += 1

        # increase index by length of entries in local stiffness matrix Kloc (9 x 9)
        index += 36

    # Loop over all membrane elements
    for el in range(nelemsMem):

        # determine strain
        membrane3DStrain(X, Y, Z, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], allDofMem, ELocal)

        # constitutive matrix
        CmatStVenantIsotropic(EMem, nu, Cmat)

        # determine stressVoigt, S1, S2, theta
        membrane3DStress(Cmat, ELocal, SLocal, &S1, &S2, &theta)

        # determine Bmat and local stress s
        membrane3DBmat(X, Y, Z, NMem, J11Vec[el], J22Vec[el], J12Vec[el], el, SLocal, s, BmatLocal)

        # determine KMem
        membrane3DKmat(X, Y, Z, NMem, BmatLocal, Cmat, KMem, s, t, areaVec[el], p[el], el)

        # local internal force vector
        dot_mv(BmatLocal.T, SLocal, FintMem)

        # row major flattening
        runner = 0
        for i in range(0, 9):

            # add local to global RHS
            Fint[allDofMem[i]] += FintMem[i] * areaVec[el] * t

            for j in range(0, 9):
                row[index + runner] = allDofMem[i]
                col[index + runner] = allDofMem[j]
                data[index + runner] = KMem[i, j]

                runner += 1

        # increase index by length of entries in local stiffness matrix Kloc (9 x 9)
        index += 81

    # Assemble sparse matrix K2D
    K = coo_matrix((data, (row, col)), shape=(ndof, ndof)).tocsc()
    K.eliminate_zeros()

    return K
