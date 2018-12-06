import numpy as np
cimport numpy as np
cimport cython

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from src.ceygen.math cimport subtract_vv
from src.ceygen.math cimport multiply_vs

from src.assembler.RHS cimport RHS2D
from src.assembler.assembleK2D cimport assembleK2D
from src.writeOutput cimport writeVTK2D

from src.helper.dirichletHandler cimport dirichlet_zero_matrix_modification
from src.helper.dirichletHandler cimport correctBC3D


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int static2D(object data) except -1:

    # type def variables
    cdef:

        int [:, ::1] NMem = data.Nm
        int [:, ::1] NBar = data.Nb

        unsigned int [:, ::1] loadedBCNodes = data.loadedBCNodes
        unsigned int [:, ::1] loadedBCEdges = data.loadedBCEdges

        double [:] Fint =  np.zeros(data.ndof, dtype=np.double)
        double [:] R =     np.ones(data.ndof, dtype=np.double)
        double [:] RHS =   np.zeros(data.ndof, dtype=np.double)
        double [:] u =     np.zeros(data.ndof, dtype=np.double)

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] X =  data.X
        double [:] Y =  data.Y

        double [:] J11Vec = data.J11Vec
        double [:] J12Vec = data.J12Vec
        double [:] J22Vec = data.J22Vec

        double [:] areaVec =  data.areaVec
        double [:] thetaVec = data.thetaVec

        double [:] L0 = data.LBar

        double [:] g = np.array([data.load["gX"], data.load["gY"]], dtype=np.double)

        double [:] RF = np.ones(len(data.dofFixed), dtype=np.double)

        unsigned int [:] row = np.empty(data.nelemsMem * 36 + data.nelemsBar * 16, dtype=np.uintc)
        unsigned int [:] col = np.empty(data.nelemsMem * 36 + data.nelemsBar * 16, dtype=np.uintc)
        double [:] dataK = np.empty(data.nelemsMem * 36 + data.nelemsBar * 16, dtype=np.double)

        unsigned int nelemsMem = data.nelemsMem
        unsigned int nelemsBar = data.nelemsBar
        unsigned int nnodes =    data.nnodes
        unsigned int ndof =      data.ndof

        unsigned int nLoadSteps = data.solverOptions["nLoadSteps"]
        unsigned int maxIter = data.solverOptions["maxIter"]
        unsigned int convergence

        unsigned int i, df, iteration = 0

        int [:] dofFixed = data.dofFixed

        int gravity

        double t =       data.props["t"]
        double areaBar = data.props["areaBar"]
        double rhoMem =  data.props["rhoMem"]
        double rhoBar =  data.props["rhoBar"]
        double EMem =    data.props["EMem"]
        double EBar =    data.props["EBar"]
        double nu =      data.props["nu"]

        double epsilonNR = data.solverOptions["epsilonNR"]

        double loadStep

        object load = data.load

    # check if gravity is active
    if data.gravity is False:
        gravity = 0
    else:
        gravity = 1

    # linear analysis
    if data.solverOptions["NL"] is False:

        # assemble RHS2D
        RHS2D(X, Y, NMem, NBar, RHS, areaVec, L0, gravity, nelemsMem, nelemsBar,
                  t, rhoMem, areaBar, rhoBar, g, loadedBCNodes, loadedBCEdges, load, 1)

        # assemble data, row, and col
        assembleK2D(NMem, NBar, X, Y, J11Vec, J22Vec, J12Vec, areaVec, L0, Fint, dataK, row, col,
                    EMem, nu, t, EBar, areaBar, nelemsMem, nelemsBar, 0)

        # Assemble sparse matrix K2D
        K = coo_matrix((dataK, (row, col)), shape=(ndof, ndof)).tocsc()
        K.eliminate_zeros()

        # correct for dirichlet BC
        K = dirichlet_zero_matrix_modification(K, dofFixed)
        correctBC3D(RHS, dofFixed)

        # solve
        u = spsolve(K, RHS)

        # update configuration
        for i in range(nnodes):
            X[i] = X0[i] + u[2 * (i + 1) - 2]
            Y[i] = Y0[i] + u[2 * (i + 1) - 1]

        # assemble Fint
        # assembleK2D(NMem, NBar, X, Y, J11Vec, J22Vec, J12Vec, areaVec, L0, Fint, dataK, row, col,
        #             EMem, nu, t, EBar, areaBar, nelemsMem, nelemsBar, 1)
        #
        # # reaction forces
        # for i in range(len(dofFixed)):
        #     RF[i] = Fint[dofFixed[i]]

    # Newton-Raphson
    else:

        loadSteps = (1 - np.linspace(0, 1, nLoadSteps, endpoint=False))[::-1]

        for loadStep in loadSteps:

            print("Load step = {}".format(loadStep))

            # assemble RHS
            RHS2D(X, Y, NMem, NBar, RHS, areaVec, L0, gravity, nelemsMem, nelemsBar,
                  t, rhoMem, areaBar, rhoBar, g, loadedBCNodes, loadedBCEdges, load, loadStep)

            # correctBC3D(RHS, dofFixed)

            # norm of RHS2D
            normRHS = np.linalg.norm(RHS)

            # scale RHS2D with loadStep
            multiply_vs(RHS, loadStep, RHS)

            # solve linear system of equations
            convergence = 0

            # Start Newton-Raphson loop
            for iteration in range(maxIter):

                # assemble data, row, and col
                assembleK2D(NMem, NBar, X, Y, J11Vec, J22Vec, J12Vec, areaVec, L0, Fint, dataK, row, col,
                EMem, nu, t, EBar, areaBar, nelemsMem, nelemsBar, iteration)

                # Assemble sparse matrix K2D
                K = coo_matrix((dataK, (row, col)), shape=(ndof, ndof)).tocsc()
                K.eliminate_zeros()

                # correct for dirichlet BC
                K = dirichlet_zero_matrix_modification(K, dofFixed)

                # Residual between internal and external load vectors
                subtract_vv(Fint, RHS, R)

                # correct for Dirichlet BCs
                correctBC3D(R, dofFixed)

                # norm of residual vector
                normR = np.linalg.norm(R)

                # Print convergence
                if data.solverOptions["printConvergence"] is True:
                    print("||R||/||RHS|| = ", normR / normRHS)

                # Stopping criteria
                if normR <= (epsilonNR * normRHS) and iteration != 1:
                    convergence = 1
                    if data.solverOptions["printConvergence"] is True:
                        print("Equilibrium reached after", iteration, "iterations")
                    break

                # Reset strain energy
                strainEnergy = 0

                # Solve next equilibrium iteration
                du = -spsolve(K, R)

                # update displacements
                for i in range(ndof):
                    u[i] += du[i]

                # update configuration
                for i in range(nnodes):
                    X[i] = X0[i] + u[2 * (i + 1) - 2]
                    Y[i] = Y0[i] + u[2 * (i + 1) - 1]

            # reaction forces
            for i in range(len(dofFixed)):
                RF[i] = Fint[dofFixed[i]]

            if convergence == 0:
                print("Warning. Not converged after", iteration, "iterations...")


    # Update data object
    data.X = X
    data.Y = Y
    data.u = u
    data.RHS = RHS
    data.RF = RF

    # Write output into vtk file
    if data.autoWrite is True:
        writeVTK2D(data)

