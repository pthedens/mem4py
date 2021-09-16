# cython: language_level=3
# cython: boundcheck=False
import numpy as np
cimport numpy as np
cimport cython
import time
from libc.math cimport isnan

from assembler.M cimport assemble_M_DR
from assembler.RHS cimport assembleRHS
from helper.sparse cimport sparsityPattern
from helper.dirichletHandler cimport correctBC
from writeOutput cimport writeVTK
from ceygen.ceygenMath cimport dot_vv
from ceygen.ceygenMath cimport add_vv
from ceygen.ceygenMath cimport subtract_vv
from ceygen.ceygenMath cimport multiply_vs
from ceygen.ceygenMath cimport multiply_vv

cdef extern from "math.h":
    double fabs(double m)
    double sqrt(double m)


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int solveKDR(object data) except -1:

    # type def variables
    cdef:

        int [:, ::1] NMem = data.Nm
        int [:, ::1] NCable = data.Nc

        double [:, ::1] loadedBCNodes = data.loadedBCNodes
        double [:, ::1] loadedBCEdges = data.loadedBCEdges

        double [:] V =     np.zeros(data.ndof, dtype=np.double)
        double [:] MV =    np.zeros(data.ndof, dtype=np.double)
        double [:] MinvR = np.zeros(data.ndof, dtype=np.double)
        double [:] Fint =  np.zeros(data.ndof, dtype=np.double)
        double [:] R =     np.empty(data.ndof, dtype=np.double)
        double [:] RHS =   np.zeros(data.ndof, dtype=np.double)
        double [:] RHS0 =  np.zeros(data.ndof, dtype=np.double)
        double [:] u =     data.u
        double [:] Minv =  np.zeros(data.ndof, dtype=np.double)
        double [:] M =     np.zeros(data.ndof, dtype=np.double)
        double [:] diagK = np.zeros(data.ndof, dtype=np.double)
        double [:] sumRowWithDiag = np.zeros(data.ndof, dtype=np.double)

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] Z0 = data.Z0

        double [:] ELocal = np.zeros(3, dtype=np.double)
        double [:] SLocal = np.zeros(3, dtype=np.double)

        double [:] area3 =  data.area3
        double [:] thetaVec = data.thetaVec
        double [:] X = data.X
        double [:] Y = data.Y
        double [:] Z = data.Z

        long double [:] J11Vec = data.J11Vec
        long double [:] J22Vec = data.J22Vec
        long double [:] J12Vec = data.J12Vec

        double [:] L0 = data.LCable

        double [:] g = np.asarray(data.load["g"], dtype=np.double)
        double [:] p = data.p
        double [:] p_fp = np.zeros((len(p)), dtype=np.double)

        # distributed load from OpenFOAM
        double [:] Sx = data.Sx
        double [:] Sy = data.Sy
        double [:] Sz = data.Sz
        double [:] pFSI = data.pFSI

        double [:] RF = np.ones(len(data.dofFixed), dtype=np.double)

        unsigned int [:] elPressurised = data.elPressurised
        unsigned int [:] elFSI = data.elFSI

        unsigned int nelemsMem = data.nelemsMem
        unsigned int nelemsCable = data.nelemsCable
        unsigned int nnodes =    data.nnodes
        unsigned int ndof =      data.ndof
        unsigned int dim =       data.dim

        unsigned int nPressurised = data.nPressurised
        unsigned int nFSI = data.nFSI

        unsigned int nLoadSteps = data.solverOptions["nLoadSteps"]
        unsigned int alphaConstant = data.solverOptions["alphaConstant"]
        unsigned int wrinklingFlag = 0

        unsigned int iteration = 0, RHS0flag = 0, writeIter = 1

        Py_ssize_t i

        int [:] dofFixed = data.dofFixed

        int gravity

        double [:] t = data.t
        double [:] area2 = data.area2
        double [:] rho3 = data.rho3
        double [:] rho2 = data.rho2
        double [:] E3 = data.E3
        double [:] E2 = data.E2
        double [:] nu = data.nu

        double epsilonKE = data.solverOptions["epsilon_KE"]
        double epsilonR =  data.solverOptions["epsilon_R"]

        double alpha = 1, loadStep

        long double KE = 1, KEOld = 0, KEOldOld = 0
        double IE = 1

        unsigned int maxIterDR = data.solverOptions["maxIterDR"]
        unsigned int outerIter = 0

        # state vector (taut = 2, wrinkled = 1, slack = 0)
        unsigned int [:] state = np.ones(nelemsMem, dtype=np.uintc) * 2

        double lam = data.solverOptions["lam"]
        str method = data.solverOptions["method"]

        double [:] force_vector = data.force_vector

        double [:] P = np.ones(nelemsMem, dtype=np.double)
        double [:] alpha_array = np.zeros(nelemsMem, dtype=np.double)

        unsigned int iter_goal = data.solverOptions["iter_goal"]
        double sigma_max = data.solverOptions["sigma_max"]
        unsigned int wrinkling_iter = 0

        double [:] pre_stress_cable = data.pre_stress_cable
        double [:] pre_strain_cable = data.pre_strain_cable

        double [:, ::1] pre_stress_membrane = data.pre_stress_membrane
        double [:, ::1] pre_strain_membrane = data.pre_strain_membrane

        double [:] pre_u = data.pre_u
        int [:] pre_u_dof = data.pre_u_dof

        unsigned int [:] pre_active = data.pre_active

    # check if gravity is active
    if data.gravity is False:
        gravity = 0
    elif data.gravity is True:
        gravity = 1

    loadSteps = (1 - np.linspace(0, 1, nLoadSteps, endpoint=False))[::-1]

    # determine sparsity pattern
    orderTemp, indptrTemp, indicesTemp = sparsityPattern(NCable, NMem, nelemsCable, nelemsMem, ndof, dim)

    cdef:
        unsigned int [:] order = np.asarray(orderTemp, dtype=np.uintc)
        unsigned int [:] indptr = np.asarray(indptrTemp, dtype=np.uintc)
        unsigned int lenData = len(indicesTemp)

        double [:] dataK =  np.zeros(lenData, dtype=np.double)
        double beta_visc = data.solverOptions["beta_visc"]
        unsigned int n_peaks = 0, ind, load_step

    KEVec = np.zeros(1)
    KEpeak = np.zeros(1, dtype=np.int)

    wrinkling_model = data.solverOptions["wrinkling_model"]
    follower_pressure = data.solverOptions["follower_pressure"]
    if data.dim == 2:
        follower_pressure = False

    if data.silent is False:
        print("""
#####################################################################################
# Running {}.msh with {} elements and {} degrees of freedom
# Using the following solver options:
# wrinkling model: {}
# mass scaling method: {}
# epsilon_KE: {:.1E}
# epsilon_R: {:.1E}
# max number of peaks: {}
# follower_pressure: {}
# number of load steps: {}
#####################################################################################""".format(data.inputName,
                                                                                                nelemsCable + nelemsMem,
                                                                                                ndof,
                                                                                                wrinkling_model,
                                                                                                method,
                                                                                                epsilonKE,
                                                                                                epsilonR,
                                                                                                maxIterDR,
                                                                                                follower_pressure,
                                                                                                nLoadSteps))

    R[...] = 1
    RF[...] = 1
    RHS[...] = 1

    RHS0[...] = 0
    KE = 1
    IE = 1

    loadStep = 0.

    cdef unsigned int n_loadsteps = 0

    # RHS vector in current configuration
    assembleRHS(X, Y, Z, pre_u, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0, gravity,
                nelemsMem, nelemsCable, nPressurised, nFSI, t, rho3, area2, rho2, g, Sx, Sy, Sz,
                pFSI, loadedBCNodes, loadedBCEdges, 0, 1 / data.solverOptions["nLoadSteps"], dim, force_vector, E2,
                pre_stress_cable, pre_strain_cable, pre_stress_membrane, pre_strain_membrane,
                pre_active, J11Vec, J22Vec, J12Vec, thetaVec, E3, nu)

    RHS0flag = 1

    # start KDR
    while np.linalg.norm(R) / np.linalg.norm(RHS) > epsilonR and \
        KE / IE > epsilonKE and \
        outerIter < maxIterDR:

        ini = True

        if np.sum(u) != 0 and data.silent is False:

            print("Time step = {}, Peak = {}, ||R|| / ||RHS|| = {:.2E} and KE / IE = {:.2E}".format(iteration,
                                                                                                    n_peaks,
                                                                                                    np.linalg.norm(R) / np.linalg.norm(RHS),
                                                                                                    KE / IE))

        KE = 0

        if n_loadsteps < data.solverOptions["nLoadSteps"]:
            loadStep += 1 / data.solverOptions["nLoadSteps"]

            # RHS vector in current configuration
            assembleRHS(X, Y, Z, pre_u, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0, gravity,
                        nelemsMem, nelemsCable, nPressurised, nFSI, t, rho3, area2, rho2, g, Sx, Sy, Sz,
                        pFSI, loadedBCNodes, loadedBCEdges, 0, loadStep, dim, force_vector, E2,
                        pre_stress_cable, pre_strain_cable, pre_stress_membrane, pre_strain_membrane,
                        pre_active, J11Vec, J22Vec, J12Vec, thetaVec, E3, nu)

            if follower_pressure is True:
                for i in range(len(p)):
                    p_fp[i] = p[i]

            n_loadsteps += 1

            if data.silent is False:
                print("loadStep = {}".format(loadStep))

        outerIter += 1

        while True:

            if ini is True:

                assemble_M_DR(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
                              nelemsMem, nelemsCable, ndof, nPressurised, nFSI, E3, E2, nu,
                              t, area2, thetaVec, area3, L0, Fint, p_fp, pFSI, order,
                              indptr, elPressurised, elFSI, state, dataK, diagK,
                              &alpha, sumRowWithDiag, dim, &IE, method, lam, P,
                              alpha_array, wrinkling_iter, iter_goal, sigma_max, loadStep,
                              ELocal, SLocal)

                # Determine residual R at current time step from internal force vector (global)
                subtract_vv(RHS, Fint, R)

                # Correct for boundary conditions
                correctBC(R, dofFixed)

                # correct for precribed displacements
                if pre_u_dof[0] != -1:

                    # set residual to zero
                    for i in pre_u_dof:
                        R[i] = 0

                # Set half step velocity to 0.5 * Minv * R0 such that V0 = 0 (Equation 62)
                multiply_vs(multiply_vv(Minv, R, V), 0.5, V)

                ini = False

            else:

                if follower_pressure is True:
                    # RHS vector in current configuration
                    assembleRHS(X, Y, Z, pre_u, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0, gravity,
                                nelemsMem, nelemsCable, nPressurised, nFSI, t, rho3, area2, rho2, g, Sx, Sy, Sz,
                                pFSI, loadedBCNodes, loadedBCEdges, 0, loadStep, dim, force_vector, E2,
                                pre_stress_cable, pre_strain_cable, pre_stress_membrane, pre_strain_membrane,
                                pre_active, J11Vec, J22Vec, J12Vec, thetaVec, E3, nu)

                assemble_M_DR(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
                              nelemsMem, nelemsCable, ndof, nPressurised, nFSI, E3, E2, nu,
                              t, area2, thetaVec, area3, L0, Fint, p_fp, pFSI, order,
                              indptr, elPressurised, elFSI, state, dataK, diagK,
                              &alpha, sumRowWithDiag, dim, &IE, method, lam, P,
                              alpha_array, wrinkling_iter, iter_goal, sigma_max, loadStep,
                              ELocal, SLocal)

                # Determine residual R at current time step from internal force vector (global)
                subtract_vv(RHS, Fint, R)

                # Correct for boundary conditions
                correctBC(R, dofFixed)

                # correct for precribed displacements
                if pre_u_dof[0] != -1:

                    # set residual to zero
                    for i in pre_u_dof:
                        R[i] = 0

                # compute nodal velocities
                add_vv(V, multiply_vs(multiply_vv(Minv, R, MinvR), 0.5, MinvR), V)

            # compute nodal displacements
            add_vv(u, V, u)

            # write displacements into position vector
            if dim == 2:
                for i in range(nnodes):
                    X[i] = X0[i] + u[2 * (i + 1) - 2]
                    Y[i] = Y0[i] + u[2 * (i + 1) - 1]
            elif dim == 3:
                for i in range(nnodes):
                    X[i] = X0[i] + u[3 * (i + 1) - 3]
                    Y[i] = Y0[i] + u[3 * (i + 1) - 2]
                    Z[i] = Z0[i] + u[3 * (i + 1) - 1]

            # Determine kinetic energy and update previous KE values
            KEOldOld = KEOld
            KEOld = KE
            KE = 0.5 * dot_vv(V, multiply_vv(M, V, MV))

            if isnan(KE):
                print('M == 0 = {}'.format(np.where(np.asarray(M) == 0)[0]))
                raise Exception("Kinetic energy blew up to infinity in iteration {}.".format(iteration))

            iteration += 1

            if KE < KEOld:

                ini = True
                wrinkling_iter += 1
                if wrinkling_iter > iter_goal and wrinkling_model == "Jarasjarungkiat":
                    print("Keeping constant stress state")

                # delta t
                alpha = (KEOld - KE) / (2 * KEOld - KEOldOld - KE)

                # update displacements
                add_vv(u, add_vv(multiply_vs(V, -(alpha + 0.5), V),
                          multiply_vs(multiply_vv(Minv, R, MV), 
                            ((1 + alpha) / 4.), MV), MinvR), u)

                # write displacements into position vector
                if dim == 2:
                    for i in range(nnodes):
                        X[i] = X0[i] + u[2 * (i + 1) - 2]
                        Y[i] = Y0[i] + u[2 * (i + 1) - 1]
                elif dim == 3:
                    for i in range(nnodes):
                        X[i] = X0[i] + u[3 * (i + 1) - 3]
                        Y[i] = Y0[i] + u[3 * (i + 1) - 2]
                        Z[i] = Z0[i] + u[3 * (i + 1) - 1]

                assemble_M_DR(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
                              nelemsMem, nelemsCable, ndof, nPressurised, nFSI, E3, E2, nu,
                              t, area2, thetaVec, area3, L0, Fint, p_fp, pFSI, order,
                              indptr, elPressurised, elFSI, state, dataK, diagK,
                              &alpha, sumRowWithDiag, dim, &IE, method, lam, P,
                              alpha_array, wrinkling_iter, iter_goal, sigma_max, loadStep,
                              ELocal, SLocal)
                
                if follower_pressure is True:
                    # RHS vector in current configuration
                    assembleRHS(X, Y, Z, pre_u, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0, gravity,
                                nelemsMem, nelemsCable, nPressurised, nFSI, t, rho3, area2, rho2, g, Sx, Sy, Sz,
                                pFSI, loadedBCNodes, loadedBCEdges, 0, loadStep, dim, force_vector, E2,
                                pre_stress_cable, pre_strain_cable, pre_stress_membrane, pre_strain_membrane,
                                pre_active, J11Vec, J22Vec, J12Vec, thetaVec, E3, nu)

                # Determine residual R at current time step from internal force vector (global)
                subtract_vv(RHS, Fint, R)

                # Correct for boundary conditions
                correctBC(R, dofFixed)

                correctBC(RHS, dofFixed)

                # Reaction forces (negative because it was defined negative previously)
                for i in range(len(dofFixed)):
                    RF[i] = Fint[dofFixed[i]]

                KEOld = 0
                KEOldOld = 0

                V[...] = 0

                n_peaks += 1

                break

    # if data.silent is False:
    if outerIter < maxIterDR:
        try:
            import colorama
            from colorama import Fore, Style
            print(Fore.GREEN + "DR converged with ||R|| / ||RHS|| = {:.2E} and KE / IE = {:.2E}".format(np.linalg.norm(R) / np.linalg.norm(RHS), KE / IE))
            print(Fore.GREEN + "after {} time steps".format(iteration))
            print(Style.RESET_ALL)
        except ImportError:
            print("DR converged with ||R|| / ||RHS|| = {:.2E} and KE / IE = {:.2E}".format(np.linalg.norm(R) / np.linalg.norm(RHS), KE / IE))
            print("after {} time steps".format(iteration))
    else:
        try:
            import colorama
            from colorama import Fore, Style
            print(Fore.RED + "DR not converged after {} time steps, ||R|| / ||RHS|| = {:.2E} and KE / IE = {:.2E}".format(iteration, np.linalg.norm(R) / np.linalg.norm(RHS), KE / IE))
            print(Style.RESET_ALL)
        except ImportError:
            print("DR not converged after {} time steps, ||R|| / ||RHS|| = {:.2E} and KE / IE = {:.2E}".format(iteration, np.linalg.norm(R) / np.linalg.norm(RHS), KE / IE))

    # Update data object
    data.X = X
    data.Y = Y
    data.Z = Z
    data.u = u
    data.V = V
    data.RHS = RHS
    data.RF = RF
    data.Fint = Fint
    data.R = R
    data.P = P
    data.alpha_array = alpha_array
    data.KEVec = KEVec
    data.KEpeak = KEpeak
    data.KE = KE
    data.M = M

    # Write output into vtk file
    if data.autoWrite is True:
        writeVTK(data)

    # TODO: OLD IMPLEMENTATION, kinda messy
    # # start DR
    # for load_step, loadStep in enumerate(loadSteps):
    #
    #     if data.silent is False:
    #         print("Load step = {}".format(load_step))
    #
    #     # set gravity load to zero
    #     RHS0[...] = 0
    #
    #     # RHS vector in current configuration
    #     assembleRHS(X, Y, Z, pre_u, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0, gravity,
    #                 nelemsMem, nelemsCable, nPressurised, nFSI, t, rho3, area2, rho2, g, Sx, Sy, Sz,
    #                 pFSI, loadedBCNodes, loadedBCEdges, 0, loadStep, dim, force_vector, E2,
    #                 pre_stress_cable, pre_strain_cable, pre_stress_membrane, pre_strain_membrane,
    #                 pre_active, J11Vec, J22Vec, J12Vec, thetaVec, E3, nu)
    #
    #     # if gravity is on it has been computed already
    #     RHS0flag = 1
    #
    #     if wrinkling_model == "Jarasjarungkiat":
    #         assembleMIsoWrinkleJarasjarungkiat(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
    #                                            nelemsMem, nelemsCable, ndof, nPressurised, E3, E2, nu,
    #                                            t, area2, thetaVec, area3, L0, Fint, p, pFSI, order,
    #                                            indptr, elPressurised, elFSI, state, dataK, diagK,
    #                                            &alpha, sumRowWithDiag, dim, &IE, method, lam, P,
    #                                            alpha_array, wrinkling_iter, iter_goal, sigma_max, loadStep)
    #     elif wrinkling_model == "Raible":
    #         assembleMIsoWrinkleRaible(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
    #                                   nelemsMem, nelemsCable, ndof, nPressurised, nFSI, E3, E2, nu,
    #                                   t, area2, thetaVec, area3, L0, Fint, Ew, p, pFSI, order,
    #                                   indptr, elPressurised, elFSI, state, dataK, diagK,
    #                                   &alpha, sumRowWithDiag, dim, 1, &IE, method, lam, V, beta_visc, loadStep)
    #     else:
    #         assembleMIso(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec, nelemsMem,
    #                      nelemsCable, ndof, nPressurised, E3, E2, nu, t, area2, thetaVec,
    #                      area3, L0, Fint, p, pFSI, order, indptr, elPressurised, elFSI,
    #                      dataK, diagK, &alpha, sumRowWithDiag, dim, method, lam, loadStep)
    #
    #     # Determine residual R at current time step from internal force vector (global)
    #     subtract_vv(RHS, Fint, R)
    #
    #     # Correct for boundary conditions
    #     correctBC(R, dofFixed)
    #     correctBC(RHS, dofFixed)
    #
    #     if np.sum(R) == 0:
    #         R[...] = 1
    #
    #     # Reaction forces (negative because it was defined negative previously)
    #     for i in range(len(dofFixed)):
    #         RF[i] = Fint[dofFixed[i]]
    #
    #     if np.sum(RF) == 0:
    #         RF[...] = 1
    #
    #     iteration = 0
    #
    #     initial_peak = 0
    #     alpha = 1
    #
    #     KE = 1
    #     KEOld = 0
    #     KEOldOld = 0
    #
    #     outerIter = 0
    #     n_peaks = 0
    #
    #     while (np.max(np.abs(R)) / np.max(np.abs(RF))) > epsilonR and \
    #             KE / IE > epsilonKE and \
    #             outerIter < maxIterDR:
    #
    #         # counter for outer iterations
    #         outerIter += 1
    #         wrinkling_iter = 0
    #
    #         if np.sum(u) != 0 and data.silent is False:
    #
    #             RoverRF[outerIter - 1] = np.max(np.abs(R)) / np.max(np.abs(RF))
    #             KEoverIE[outerIter - 1] = KE / IE
    #
    #             print("Time step = {}, Peak = {}, max(|R|) / max(|RF|) = {:.2E} and KE / IE = {:.2E}".format(iteration,
    #                                                                                            n_peaks,
    #                                                                                            np.max(np.abs(R)) / np.max(np.abs(RF)),
    #                                                                                            KE / IE))
    #             # if dim == 2:
    #             #     print("max(X) = {:.2E}, max(Y) = {:.2E}".format(max(X), max(Y)))
    #             # elif dim == 3:
    #             #     print("max(X) = {:.2E}, max(Y) = {:.2E}, max(Z) = {:.2E}".format(max(X), max(Y), max(Z)))
    #
    #         # RHS vector in current configuration
    #         assembleRHS(X, Y, Z, pre_u, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0, gravity,
    #                     nelemsMem, nelemsCable, nPressurised, nFSI, t, rho3, area2, rho2, g, Sx, Sy, Sz,
    #                     pFSI, loadedBCNodes, loadedBCEdges, 0, loadStep, dim, force_vector, E2,
    #                     pre_stress_cable, pre_strain_cable, pre_stress_membrane, pre_strain_membrane,
    #                     pre_active, J11Vec, J22Vec, J12Vec, thetaVec, E3, nu)
    #
    #         # Set old kinetic energy to zero
    #         KEOld = 0
    #         KEOldOld = 0
    #
    #         # Correct for boundary conditions
    #         correctBC(RHS, dofFixed)
    #
    #         # Set half step velocity to 0.5 * Minv * R0 such that V0 = 0 (Equation 62)
    #         multiply_vs(multiply_vv(Minv, R, V), (0.5 * alpha), V)
    #
    #         correctBC(V, dofFixed)
    #
    #         # reset wrinkling strains
    #         for i in range(nelemsMem):
    #             Ew[i, 0] = 0
    #             Ew[i, 1] = 0
    #             Ew[i, 2] = 0
    #
    #         while True:
    #
    #             if wrinkling_model == "Jarasjarungkiat":
    #                 assembleMIsoWrinkleJarasjarungkiat(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
    #                                                    nelemsMem, nelemsCable, ndof, nPressurised, E3, E2, nu,
    #                                                    t, area2, thetaVec, area3, L0, Fint, p, pFSI, order,
    #                                                    indptr, elPressurised, elFSI, state, dataK, diagK,
    #                                                    &alpha, sumRowWithDiag, dim, &IE, method, lam, P,
    #                                                    alpha_array, wrinkling_iter, iter_goal, sigma_max, loadStep)
    #             elif wrinkling_model == "Raible":
    #                 assembleMIsoWrinkleRaible(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
    #                                           nelemsMem, nelemsCable, ndof, nPressurised, nFSI, E3, E2, nu,
    #                                           t, area2, thetaVec, area3, L0, Fint, Ew, p, pFSI, order,
    #                                           indptr, elPressurised, elFSI, state, dataK, diagK,
    #                                           &alpha, sumRowWithDiag, dim, 1, &IE, method, lam, V, beta_visc, loadStep)
    #             else:
    #                 assembleMIso(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec, nelemsMem,
    #                              nelemsCable, ndof, nPressurised, E3, E2, nu, t, area2, thetaVec,
    #                              area3, L0, Fint, p, pFSI, order, indptr, elPressurised, elFSI,
    #                              dataK, diagK, &alpha, sumRowWithDiag, dim, method, lam, loadStep)
    #
    #             if initial_peak >= 1:
    #                 iter_goal = data.solverOptions["iter_goal"]
    #
    #             alpha = 1
    #
    #             if np.fmod(iteration, 1000) == 0 and data.silent is False:
    #                 print("Time step = {} at {}".format(iteration, time.asctime( time.localtime(time.time()) )))
    #
    #             iteration += 1
    #
    #             wrinkling_iter += 1
    #
    #             # Determine residual R at current time step from internal force vector (global)
    #             subtract_vv(RHS, Fint, R)
    #
    #             # Correct for boundary conditions
    #             correctBC(R, dofFixed)
    #             correctBC(RHS, dofFixed)
    #
    #             # Update velocity in current configuration at time t+dt/2 (Equation 5)
    #             add_vv(V, multiply_vs(multiply_vv(Minv, R, MinvR), ((alpha + 1) / 2.), MinvR), V)
    #
    #             correctBC(V, dofFixed)
    #
    #             # correct for precribed displacements
    #             if pre_u_dof[0] != -1:
    #                 for i in pre_u_dof:
    #                     V[i] = 0
    #
    #             # Determine kinetic energy and update previous KE values
    #             KE = 0.5 * dot_vv(V, multiply_vv(M, V, MV))
    #             KE_nomass = 0.5 * dot_vv(V, V)
    #
    #             KEVec = np.append(KEVec, KE)
    #             KEpeak = np.append(KEpeak, 0)
    #
    #             save_res_KE.append(KE_nomass)
    #             save_res_R.append(np.max(np.abs(R)))
    #
    #             add_vv(u, multiply_vs(V, alpha, MV), u)
    #
    #             # correct for precribed displacements
    #             if pre_u_dof[0] != -1:
    #
    #                 # set RHS to zero
    #                 for i in range(ndof):
    #                     RHS[i] = 0
    #
    #                 ind = 0
    #                 for i in pre_u_dof:
    #                     u[i] = pre_u[ind]
    #                     ind += 1
    #
    #             if dim == 2:
    #                 for i in range(nnodes):
    #                     X[i] = X0[i] + u[2 * (i + 1) - 2]
    #                     Y[i] = Y0[i] + u[2 * (i + 1) - 1]
    #             elif dim == 3:
    #                 for i in range(nnodes):
    #                     X[i] = X0[i] + u[3 * (i + 1) - 3]
    #                     Y[i] = Y0[i] + u[3 * (i + 1) - 2]
    #                     Z[i] = Z0[i] + u[3 * (i + 1) - 1]
    #
    #             save_Z.append(Z[3])
    #
    #             if follower_pressure is True:
    #
    #                 assembleRHS(X, Y, Z, u, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0,
    #                             gravity, nelemsMem, nelemsCable, nPressurised, nFSI, t, rho3, area2,
    #                             rho2, g, Sx, Sy, Sz, pFSI, loadedBCNodes, loadedBCEdges, RHS0flag,
    #                             loadStep, dim, force_vector, E2, pre_stress_cable, pre_strain_cable,
    #                             pre_stress_membrane, pre_strain_membrane, pre_active, J11Vec, J22Vec,
    #                             J12Vec, thetaVec, E3, nu)
    #
    #                 correctBC(RHS, dofFixed)
    #
    #             if isnan(KE):
    #                 raise Exception("Kinetic energy blew up to infinity in iteration {}.".format(iteration))
    #
    #             if data.write_each_timestep is True:
    #                 # Update data object
    #                 data.X = X
    #                 data.Y = Y
    #                 data.Z = Z
    #                 data.u = u
    #                 data.V = V
    #                 data.RHS = RHS
    #                 data.Fint = Fint
    #                 data.Ew = Ew
    #                 data.RF = RF
    #                 data.R = R
    #                 data.time = writeIter
    #                 writeIter += 1
    #
    #                 data.P = P
    #                 data.alpha_array = alpha_array
    #
    #                 data.KE = KE
    #                 data.M = M
    #
    #                 # Write output into vtk file
    #                 writeVTK(data)
    #
    #             if initial_peak >= 1:
    #                 iter_goal = data.solverOptions["iter_goal"]
    #
    #             # check if peak in kinetic energy occurred
    #             if KE - KEOld < 0:
    #
    #                 KEpeak[iteration] = 1
    #
    #                 # KE peak flag
    #                 initial_peak += 1
    #
    #                 n_peaks += 1
    #
    #                 # alpha = (KEOld - KE) / (2 * KEOld - KEOldOld - KE)
    #
    #                 # Update nodal position to current time step (Equation 61)
    #                 add_vv(u, add_vv(multiply_vs(V, -(alpha + 0.5), V),
    #                                  multiply_vs(multiply_vv(Minv, R, MV),
    #                                              ((1 + alpha) / 4.), MV), MinvR), u)
    #
    #                 correctBC(u, dofFixed)
    #
    #                 # correct for prescribed displacements
    #                 if pre_u_dof[0] != -1:
    #                     ind = 0
    #                     for i in pre_u_dof:
    #                         u[i] = pre_u[ind]
    #                         ind += 1
    #
    #                 if dim == 2:
    #                     for i in range(nnodes):
    #                         X[i] = X0[i] + u[2 * (i + 1) - 2]
    #                         Y[i] = Y0[i] + u[2 * (i + 1) - 1]
    #                 elif dim == 3:
    #                     for i in range(nnodes):
    #                         X[i] = X0[i] + u[3 * (i + 1) - 3]
    #                         Y[i] = Y0[i] + u[3 * (i + 1) - 2]
    #                         Z[i] = Z0[i] + u[3 * (i + 1) - 1]
    #
    #                 if wrinkling_model == "Jarasjarungkiat":
    #                     assembleMIsoWrinkleJarasjarungkiat(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
    #                                                            nelemsMem, nelemsCable, ndof, nPressurised, E3, E2, nu,
    #                                                            t, area2, thetaVec, area3, L0, Fint, p, pFSI, order,
    #                                                            indptr, elPressurised, elFSI, state, dataK, diagK,
    #                                                            &alpha, sumRowWithDiag, dim, &IE, method, lam, P,
    #                                                            alpha_array, 2, 1, sigma_max, loadStep)
    #                 elif wrinkling_model == "Raible":
    #                     assembleMIsoWrinkleRaible(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec,
    #                                               nelemsMem, nelemsCable, ndof, nPressurised, nFSI, E3, E2, nu,
    #                                               t, area2, thetaVec, area3, L0, Fint, Ew, p, pFSI, order,
    #                                               indptr, elPressurised, elFSI, state, dataK, diagK,
    #                                               &alpha, sumRowWithDiag, dim, 1, &IE, method, lam, V, beta_visc, loadStep)
    #                 else:
    #                     assembleMIso(M, Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec, nelemsMem,
    #                                  nelemsCable, ndof, nPressurised, E3, E2, nu, t, area2, thetaVec,
    #                                  area3, L0, Fint, p, pFSI, order, indptr, elPressurised, elFSI,
    #                                  dataK, diagK, &alpha, sumRowWithDiag, dim, method, lam, loadStep)
    #
    #                 if alphaConstant == 1:
    #                     alpha = 1
    #
    #                 # Determine residual R at current time step from internal force vector (global)
    #                 subtract_vv(RHS, Fint, R)
    #
    #                 # Correct for boundary conditions
    #                 correctBC(R, dofFixed)
    #
    #                 # Reaction forces (negative because it was defined negative previously)
    #                 for i in range(len(dofFixed)):
    #                     RF[i] = Fint[dofFixed[i]]
    #
    #                 # Determine kinetic energy and update previous KE values
    #                 KE = 0.5 * dot_vv(V, multiply_vv(M, V, MV))
    #
    #                 if data.write_each_peak is True:
    #                     # Update data object
    #                     data.X = X
    #                     data.Y = Y
    #                     data.Z = Z
    #                     data.u = u
    #                     data.V = V
    #                     data.RHS = RHS
    #                     data.Fint = Fint
    #                     data.Ew = Ew
    #                     data.RF = RF
    #                     data.R = R
    #                     data.time = writeIter
    #                     writeIter += 1
    #
    #                     data.P = P
    #                     data.alpha_array = alpha_array
    #
    #                     data.KE = KE
    #                     data.M = M
    #
    #                     # Write output into vtk file
    #                     writeVTK(data)
    #
    #                 # internal energy
    #                 # IE = dot_vv(Fint, u)
    #
    #                 # strain energy
    #                 data.strainEnergy = IE
    #
    #                 # Update old kinetic energy value
    #                 KEOld = 0
    #
    #                 break
    #
    #             # Update old kinetic energy value
    #             KEOld = KE
    #
    #             # internal energy
    #             # IE = dot_vv(Fint, u)
    #
    #             # strain energy
    #             data.strainEnergy = IE