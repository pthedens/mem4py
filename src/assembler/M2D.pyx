import numpy as np
cimport numpy as np
cimport cython

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import norm, eigs

from src.ceygen.math cimport dot_mm, dot_mv

from src.materialModels.StVenantIso cimport CmatStVenantIsotropic

from src.elements.bar2D cimport bar2DFintAndK

from src.elements.membrane2D cimport membrane2DStrain
from src.elements.membrane2D cimport membrane2DStress
from src.elements.membrane2D cimport membrane2DBmat
from src.elements.membrane2D cimport membrane2DKmat

cdef extern from "math.h":
    double sqrt(double m)
    double cos(double m)
    double sin(double m)
    double atan2(double m, double n)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void M2DIsotropic(double [:] M,
                               double [:] Minv,
                               double [:] X,
                               double [:] Y,
                               double [:] Fint,
                               int [:, ::1] NMem,
                               int [:, ::1] NBar,
                               double [:] J11Vec,
                               double [:] J22Vec,
                               double [:] J12Vec,
                               unsigned int nelemsMem,
                               unsigned int nelemsBar,
                               unsigned int ndof,
                               double EMem,
                               double poisson,
                               double t,
                               double EBar,
                               double areaBar,
                               double [:] L0,
                               double [:] thetaVec,
                               double [:] areaVec):

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
        unsigned int [:] allDofBar = np.zeros(4, dtype=np.uintc)

        double [:] FintBar = np.zeros(4, dtype=np.double)
        double [:, ::1] KBar = np.zeros((4, 4), dtype=np.double)

        # membrane elements
        unsigned int [:] allDofMem = np.zeros(6, dtype=np.uintc)

        double [:] SLocal =  np.zeros(3, dtype=np.double)
        double [:] s =       np.zeros(3, dtype=np.double)
        double [:] FintMem = np.zeros(6, dtype=np.double)

        double [:, :] BmatLocal = np.zeros((3, 6), dtype=np.double)
        double [:, ::1] KMem =    np.zeros((6, 6), dtype=np.double)
        double [:, :] Cmat =      np.zeros((3, 3), dtype=np.double)

        double [:] data =  np.empty(nelemsMem * 36 + nelemsBar * 16, dtype=np.double)
        double [:] diagK = np.zeros(ndof, dtype=np.double)
        
        unsigned int [:] row =  np.empty(nelemsMem * 36 + nelemsBar * 16, dtype=np.uintc)
        unsigned int [:] col =  np.empty(nelemsMem * 36 + nelemsBar * 16, dtype=np.uintc)

        unsigned int el, i, j, index = 0, runner

    Fint[...] = 0
    
    # Loop over all bar elements
    for el in range(nelemsBar):

        bar2DFintAndK(X, Y, NBar, L0[el], el, areaBar, EBar, FintBar, allDofBar, KBar)

        # row major flattening
        runner = 0
        for i in range(4):
        
            # add local to global RHS
            Fint[allDofBar[i]] += FintBar[i]

            for j in range(4):
                row[index + runner] = allDofBar[i]
                col[index + runner] = allDofBar[j]
                data[index + runner] = KBar[i, j]

                if i == j:
                    diagK[allDofBar[i]] += data[index + runner]

                runner += 1

        # increase index by length of entries in local stiffness matrix Kloc (4 x 4)
        index += 16

    # constitutive matrix
    CmatStVenantIsotropic(EMem, poisson, Cmat)

    # Loop over all elements
    for el in range(nelemsMem):

        # Find degrees of freedom from current element
        allDofMem[0] = 2 * (NMem[el, 1] + 1) - 2
        allDofMem[1] = 2 * (NMem[el, 1] + 1) - 1
        allDofMem[2] = 2 * (NMem[el, 2] + 1) - 2
        allDofMem[3] = 2 * (NMem[el, 2] + 1) - 1
        allDofMem[4] = 2 * (NMem[el, 3] + 1) - 2
        allDofMem[5] = 2 * (NMem[el, 3] + 1) - 1

        # determine Bmat and local stress s
        membrane2DBmat(X, Y, NMem, J11Vec[el], J22Vec[el], J12Vec[el], el,
                       SLocal, s, BmatLocal)

        # determine KMem
        membrane2DKmat(BmatLocal, Cmat, KMem, s, t, areaVec[el])

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
    K = coo_matrix((data, (row, col)), shape=(ndof, ndof)).tocsr()
    K.eliminate_zeros()

    cdef double [:] sumRowWithDiag = np.empty(ndof, dtype=np.double)
    cdef double [:] sumColWithDiag = np.empty(ndof, dtype=np.double)
    cdef double [:] sumRowNoDiag = np.empty(ndof, dtype=np.double)
    cdef double [:] sumColNoDiag = np.empty(ndof, dtype=np.double)
    cdef double [:] alphaVec = np.zeros(ndof, dtype=np.double)
    cdef double betaCol, betaRow, alphaSqrt, alphaRow, alphaCol, Sum

    # Summ columns without diagonal component
    sumColWithDiag = np.asarray(np.abs(K).sum(axis=0)).reshape((ndof,))
    sumRowWithDiag = np.asarray(np.abs(K).sum(axis=1)).reshape((ndof,))
    sumColNoDiag = np.asarray(sumColWithDiag) - np.abs(diagK)
    sumRowNoDiag = np.asarray(sumRowWithDiag) - np.abs(diagK)

    # loop through columns of K and determine alpha
    for i in range(ndof):

        if np.abs(diagK[i]) > 1E-18:

            betaCol = sumColNoDiag[i] / diagK[i]
            betaRow = sumRowNoDiag[i] / diagK[i]

            # check for col condition
            if diagK[i] > sumColNoDiag[i]:
                alphaCol = (1 - sqrt(1 - betaCol * betaCol)) / betaCol
            elif diagK[i] < 0.5 * sumColNoDiag[i] or np.abs(diagK[i] - sumColNoDiag[i]) < 1E-14:
                alphaCol = (betaCol + 2 - 2 * sqrt(1 + betaCol)) / betaCol
            # elif 0.5 * sumColNoDiag[i] <= diagK[i] < sumColNoDiag[i]:
            else:
                alphaCol = betaCol - sqrt(betaCol * betaCol - 1)

            # check for row condition
            if diagK[i] > sumRowNoDiag[i]:
                alphaRow = (1 - sqrt(1 - betaRow * betaRow)) / betaRow
            elif diagK[i] < 0.5 * sumRowNoDiag[i] or np.abs(diagK[i] - sumRowNoDiag[i]) < 1E-14:
                alphaRow = (betaRow + 2 - 2 * sqrt(1 + betaRow)) / betaRow
            # elif 0.5 * sumColNoDiag[i] <= diagK[i] < sumColNoDiag[i]:
            else:
                alphaRow = betaRow - sqrt(betaRow * betaRow - 1)

            alphaVec[i] = np.max([alphaRow, alphaCol])

    # find max alpha (sqrt(alpha) actually)
    alphaSqrt = np.max(alphaVec)

    # assemble mass matrix
    for i in range(ndof):

        M[i] = (1 + alphaSqrt * alphaSqrt) / (2 * (1 + alphaSqrt) ** 2) * \
               np.max([sumRowWithDiag[i], sumColWithDiag[i]])
        
        Minv[i] = 1. / M[i]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void M2DExplicit(int [:, :] N,
                      double [:] Minv,
                      double [:] areaVec,
                      double t,
                      double rho,
                      unsigned int nelems):


    cdef unsigned int [:] allDofE = np.empty(9, dtype=np.uintc)
    cdef unsigned int el, dof
    cdef double MLocal
    print("FIX M2DExplicit")
    # TODO: FIX ME
    for el in range(nelems):

        allDofE[0] = 3 * (N[el, 1] + 1) - 3
        allDofE[1] = 3 * (N[el, 1] + 1) - 2
        allDofE[2] = 3 * (N[el, 1] + 1) - 1
        allDofE[3] = 3 * (N[el, 2] + 1) - 3
        allDofE[4] = 3 * (N[el, 2] + 1) - 2
        allDofE[5] = 3 * (N[el, 2] + 1) - 1
        allDofE[6] = 3 * (N[el, 3] + 1) - 3
        allDofE[7] = 3 * (N[el, 3] + 1) - 2
        allDofE[8] = 3 * (N[el, 3] + 1) - 1

        MLocal = rho * areaVec[el] * t

        for dof in range(9):
            # TODO: ONLY VALID FOR CST ELEMENTS
            Minv[allDofE[dof]] += 3 / MLocal


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef object K2DIsotropic(double [:] X,
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

    cdef double [:, :] Tepsilon = np.zeros((3, 3), dtype=np.double)
    cdef double [:, :] Tsigma = np.zeros((3, 3), dtype=np.double)
    cdef double [:, :] CT = np.zeros((3, 3), dtype=np.double)
    cdef double [:, :] CmatTilde = np.zeros((3, 3), dtype=np.double)

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
        ELocal[2] = (g12 - g11 * J12Vec[el] / J11Vec[el]) / (J11Vec[el] * J22Vec[el]) / 2.

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

        # strain displacement matrix in curvilinear coordinate system
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

        # set BmatCurv to zero
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

            # minus to subtract from RHS for residual
            Fint[allDofEMem[i]] -= FintElement[i] * areaVec[el] * t

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

    return K
