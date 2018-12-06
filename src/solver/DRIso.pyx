import numpy as np
cimport numpy as np
cimport cython

from src.assembler.M3D cimport M3DIsotropic
from src.assembler.M2D cimport M2DIsotropic
from src.assembler.RHS cimport RHS3D, RHS2D
from src.helper.dirichletHandler cimport correctBC3D
from src.writeOutput cimport writeVTK3D, writeVTK2D

from libc.math cimport isnan
from src.ceygen.math cimport dot_vv
from src.ceygen.math cimport add_vv
from src.ceygen.math cimport subtract_vv
from src.ceygen.math cimport multiply_vs
from src.ceygen.math cimport multiply_vv


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void DRIsotropic3D(object data):

    cdef:
        # type def variables
        double [:] MV = np.zeros(data.ndof, dtype=np.double)
        double [:] MinvR = np.zeros(data.ndof, dtype=np.double)
        double [:] Fint = np.zeros(data.ndof, dtype=np.double)
        double [:] R = np.ones(data.ndof, dtype=np.double)
        double [:] RHS = np.zeros(data.ndof, dtype=np.double)
        double [:] RHS0 = np.zeros(data.ndof, dtype=np.double)
        double [:] Minv = np.zeros(data.ndof, dtype=np.double)
        double [:] M = np.zeros(data.ndof, dtype=np.double)

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] Z0 = data.Z0
        double [:] X = data.X
        double [:] Y = data.Y
        double [:] Z = data.Z
        double [:] u = data.u
        double [:] V = data.V
        int [:, ::1] NMem = data.Nm
        int [:, ::1] NBar = data.Nb

        unsigned int [:, ::1] loadedBCNodes = data.loadedBCNodes
        unsigned int [:, ::1] loadedBCEdges = data.loadedBCEdges

        double [:] L0 = data.LBar

        double [:] J11Vec = data.J11Vec
        double [:] J12Vec = data.J12Vec
        double [:] J22Vec = data.J22Vec

        double [:] areaVec = data.areaVec
        double [:] thetaVec = data.thetaVec

        unsigned int nelems = data.nelems, nnodes = data.nnodes, ndof = data.ndof
        unsigned int nelemsMem = data.nelemsMem
        unsigned int nelemsBar = data.nelemsBar

        double t =       data.props["t"]
        double areaBar = data.props["areaBar"]
        double rhoMem =  data.props["rhoMem"]
        double rhoBar =  data.props["rhoBar"]
        double EMem =    data.props["EMem"]
        double EBar =    data.props["EBar"]
        double poisson = data.props["nu"]

        object load = data.load

        double [:] g = np.array([data.load["gX"], data.load["gY"], data.load["gZ"]], dtype=np.double)

        int gravity

        double epsilonKE = data.solverOptions["epsilonKE"]
        double epsilonR =  data.solverOptions["epsilonR"]

        # Internal pressure
        double [:] p = data.p

        # distributed load from OpenFOAM
        double [:] Sx = data.Sx
        double [:] Sy = data.Sy
        double [:] Sz = data.Sz

        unsigned int i, nLoadSteps = data.solverOptions["nLoadSteps"], RHS0flag = 0
        double q, loadStep
        long double KE, KEOld, KEOldOld, IE

        unsigned int [:] elPressurised = data.elPressurised
        int [:] dofFixed = data.dofFixed

        double [:] RF = np.ones(len(dofFixed), dtype=np.double)

    if data.gravity is False:
        gravity = 0
    elif data.gravity is True:
        gravity = 1

    # Initialise kinetic energy
    KE = 1
    IE = 1

    loadSteps = (1 - np.linspace(0, 1, nLoadSteps, endpoint=False))[::-1]

    for loadStep in loadSteps:

        print("Load step = {}".format(loadStep))

        RHS0flag = 0

        # RHS vector in current configuration
        RHS3D(X, Y, Z, NMem, NBar, p, RHS, RHS0, elPressurised, areaVec, L0, gravity, nelemsMem,
              nelemsBar, t, rhoMem, areaBar, rhoBar, g, Sx, Sy, Sz, loadedBCNodes, loadedBCEdges,
              RHS0flag, load, loadStep)

        RHS0flag = 1

        R = np.copy(RHS)
        RF = np.copy(RHS)

        while (np.max(np.abs(R)) / np.max(np.abs(RF))) > epsilonR and \
                                KE / IE > epsilonKE:

            if np.sum(u) != 0:
                print("R / RF = {} and KE / IE = {}".format(np.max(np.abs(R)) / np.max(np.abs(RF)),
                                                            KE / IE))

            # Set old kinetic energy to zero
            KEOld = 0
            KEOldOld = 0


            # Correct for boundary conditions
            correctBC3D(R, dofFixed)

            # Set half step velocity to 0.5 * Minv * R0 such that V0 = 0
            # V[...] = 0
            multiply_vs(multiply_vv(Minv, R, V), 0.5, V)

            # RHS vector in current configuration
            RHS3D(X, Y, Z, NMem, NBar, p, RHS, RHS0, elPressurised, areaVec, L0, gravity, nelemsMem,
              nelemsBar, t, rhoMem, areaBar, rhoBar, g, Sx, Sy, Sz, loadedBCNodes, loadedBCEdges,
              RHS0flag, load, loadStep)

            while True:

                # Internal force vector and scaled mass matrix
                M3DIsotropic(M, Minv, X, Y, Z, NMem, J11Vec, J22Vec, J12Vec, nelems,
                             nnodes, EMem, poisson, t, thetaVec, areaVec, Fint)

                # Determine residual R at current time step from internal force vector (global)
                subtract_vv(RHS, Fint, R)

                # Correct for boundary conditions
                correctBC3D(R, dofFixed)

                # Update velocity in current configuration at time t+dt/2
                add_vv(V, multiply_vv(Minv, R, MinvR), V)

                # Determine kinetic energy and update previous KE values
                KE = 0.5 * dot_vv(V, multiply_vv(M, V, MV))
                # KEVec[bla] = KE
                # bla+= 1

                # Update displacements in current configuration at time t+dt/2
                add_vv(u, V, u)

                for i in range(nnodes):
                    X[i] = X0[i] + u[3 * (i + 1) - 3]
                    Y[i] = Y0[i] + u[3 * (i + 1) - 2]
                    Z[i] = Z0[i] + u[3 * (i + 1) - 1]

                if isnan(KE):
                    raise Exception("Kinetic energy blew up to infinity.")

                # Check for kinetic energy peak
                if KE < KEOld:

                    # Kinetic energy ratios
                    q = 0.5

                    # Update nodal position to current time step
                    add_vv(u, add_vv(multiply_vs(V, -(1 + q), V),
                                     multiply_vs(multiply_vv(Minv, R, MV),
                                                 q, MV), MinvR), u)

                    for i in range(nnodes):

                        X[i] = X0[i] + u[3 * (i + 1) - 3]
                        Y[i] = Y0[i] + u[3 * (i + 1) - 2]
                        Z[i] = Z0[i] + u[3 * (i + 1) - 1]

                    # Internal force vector and scaled mass matrix
                    M3DIsotropic(M, Minv, X, Y, Z, NMem, J11Vec, J22Vec, J12Vec, nelems,
                                 nnodes, EMem, poisson, t, thetaVec, areaVec, Fint)

                    # Determine residual R at current time step from internal force vector (global)
                    subtract_vv(RHS, Fint, R)

                    # Correct for boundary conditions
                    correctBC3D(R, dofFixed)

                    # # Reaction forces
                    for i in range(len(dofFixed)):
                        RF[i] = Fint[dofFixed[i]]

                    break

                # Update old kinetic energy value
                KEOldOld = KEOld
                KEOld = KE

                # internal energy
                IE = np.abs(dot_vv(Fint, u))

    print("DR converged with residual norm {}".format(np.max(np.abs(R)) / np.max(np.abs(RF))))

    # Update data object
    data.X = X
    data.Y = Y
    data.Z = Z
    data.u = u
    data.V = V
    data.RHS = RHS
    data.RF = RF

    # Write output into vtk file
    if data.autoWrite is True:
        writeVTK3D(data)



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void DRIsotropic2D(object data):

    cdef:
        # type def variables
        double [:] MV = np.zeros(data.ndof, dtype=np.double)
        double [:] MinvR = np.zeros(data.ndof, dtype=np.double)
        double [:] Fint = np.zeros(data.ndof, dtype=np.double)
        double [:] R = np.ones(data.ndof, dtype=np.double)
        double [:] RHS = np.zeros(data.ndof, dtype=np.double)
        double [:] RHS0 = np.zeros(data.ndof, dtype=np.double)
        double [:] Minv = np.zeros(data.ndof, dtype=np.double)
        double [:] M = np.zeros(data.ndof, dtype=np.double)

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] Z0 = data.Z0
        double [:] X = data.X
        double [:] Y = data.Y
        double [:] Z = data.Z
        double [:] u = data.u
        double [:] V = data.V
        int [:, ::1] NMem = data.Nm
        int [:, ::1] NBar = data.Nb

        unsigned int [:, ::1] loadedBCNodes = data.loadedBCNodes
        unsigned int [:, ::1] loadedBCEdges = data.loadedBCEdges

        double [:] L0 = data.LBar

        double [:] J11Vec = data.J11Vec
        double [:] J12Vec = data.J12Vec
        double [:] J22Vec = data.J22Vec

        double [:] areaVec = data.areaVec
        double [:] thetaVec = data.thetaVec

        unsigned int nelems = data.nelems, nnodes = data.nnodes, ndof = data.ndof
        unsigned int nelemsMem = data.nelemsMem
        unsigned int nelemsBar = data.nelemsBar

        double t =       data.props["t"]
        double areaBar = data.props["areaBar"]
        double rhoMem =  data.props["rhoMem"]
        double rhoBar =  data.props["rhoBar"]
        double EMem =    data.props["EMem"]
        double EBar =    data.props["EBar"]
        double poisson = data.props["nu"]

        object load = data.load

        double [:] g = np.array([data.load["gX"], data.load["gY"]], dtype=np.double)

        int gravity

        double epsilonKE = data.solverOptions["epsilonKE"]
        double epsilonR =  data.solverOptions["epsilonR"]

        unsigned int i, nLoadSteps = data.solverOptions["nLoadSteps"], RHS0flag = 0
        double q, loadStep
        long double KE, KEOld, KEOldOld, IE

        unsigned int [:] elPressurised = data.elPressurised
        int [:] dofFixed = data.dofFixed

        double [:] RF = np.ones(len(dofFixed), dtype=np.double)
    
    if data.gravity is False:
        gravity = 0
    elif data.gravity is True:
        gravity = 1

    # Initialise kinetic energy
    KE = 1
    IE = 1

    loadSteps = (1 - np.linspace(0, 1, nLoadSteps, endpoint=False))[::-1]

    for loadStep in loadSteps:

        print("Load step = {}".format(loadStep))

        RHS0flag = 0

        # RHS vector in current configuration
        RHS2D(X, Y, NMem, NBar, RHS, areaVec, L0, gravity, nelemsMem, nelemsBar, t,
              rhoMem, areaBar, rhoBar, g, loadedBCNodes, loadedBCEdges, load, RHS0flag)

        RHS0flag = 1

        R = np.copy(RHS)
        RF = np.copy(RHS)

        while (np.max(np.abs(R)) / np.max(np.abs(RF))) > epsilonR and \
                                KE / IE > epsilonKE:

            if np.sum(u) != 0:
                print("R / RF = {} and KE / IE = {}".format(np.max(np.abs(R)) / np.max(np.abs(RF)),
                                                            KE / IE))

            # Set old kinetic energy to zero
            KEOld = 0
            KEOldOld = 0


            # Correct for boundary conditions
            correctBC3D(R, dofFixed)

            # Set half step velocity to 0.5 * Minv * R0 such that V0 = 0
            multiply_vs(multiply_vv(Minv, R, V), 0.5, V)

            # RHS vector in current configuration
            # RHS2D

            while True:

                # Internal force vector and scaled mass matrix
                M2DIsotropic(M, Minv, X, Y, Fint, NMem, NBar, J11Vec, J22Vec, J12Vec,
                nelemsMem, nelemsBar, ndof, EMem, poisson, t, EBar, areaBar, L0, thetaVec,
                areaVec)

                # Determine residual R at current time step from internal force vector (global)
                subtract_vv(RHS, Fint, R)

                # Correct for boundary conditions
                correctBC3D(R, dofFixed)

                # Update velocity in current configuration at time t+dt/2
                add_vv(V, multiply_vv(Minv, R, MinvR), V)

                # Determine kinetic energy and update previous KE values
                KE = 0.5 * dot_vv(V, multiply_vv(M, V, MV))
                # KEVec[bla] = KE
                # bla+= 1

                # Update displacements in current configuration at time t+dt/2
                add_vv(u, V, u)

                for i in range(nnodes):
                    X[i] = X0[i] + u[2 * (i + 1) - 2]
                    Y[i] = Y0[i] + u[2 * (i + 1) - 1]

                if isnan(KE):
                    raise Exception("Kinetic energy blew up to infinity.")

                # Check for kinetic energy peak
                if KE < KEOld:

                    # Kinetic energy ratios
                    q = 0.5

                    # Update nodal position to current time step
                    add_vv(u, add_vv(multiply_vs(V, -(1 + q), V),
                                     multiply_vs(multiply_vv(Minv, R, MV),
                                                 q, MV), MinvR), u)

                    for i in range(nnodes):

                        X[i] = X0[i] + u[2 * (i + 1) - 2]
                        Y[i] = Y0[i] + u[2 * (i + 1) - 1]

                    # Internal force vector and scaled mass matrix
                    M2DIsotropic(M, Minv, X, Y, Fint, NMem, NBar, J11Vec, J22Vec, J12Vec,
                nelemsMem, nelemsBar, ndof, EMem, poisson, t, EBar, areaBar, L0, thetaVec,
                areaVec)

                    # Determine residual R at current time step from internal force vector (global)
                    subtract_vv(RHS, Fint, R)

                    # Correct for boundary conditions
                    correctBC3D(R, dofFixed)

                    # # Reaction forces
                    for i in range(len(dofFixed)):
                        RF[i] = Fint[dofFixed[i]]

                    break

                # Update old kinetic energy value
                KEOldOld = KEOld
                KEOld = KE

                # internal energy
                IE = np.abs(dot_vv(Fint, u))

    print("DR converged with residual norm {}".format(np.max(np.abs(R)) / np.max(np.abs(RF))))

    # Update data object
    data.X = X
    data.Y = Y
    data.u = u
    data.V = V
    data.RHS = RHS
    data.RF = RF

    # Write output into vtk file
    if data.autoWrite is True:
        writeVTK2D(data)
