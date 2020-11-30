# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
cimport cython
import time

from src.assembler.M cimport assembleM
from src.assembler.K cimport computeuDotDot
from src.assembler.RHS cimport assembleRHS
from src.helper.dirichletHandler cimport correctBC
from src.helper.sparse cimport sparsityPattern
from src.writeOutput cimport writeVTK


cdef int solveTransient(object data) except -1:

    # type def variables
    cdef:

        int [:, ::1] NMem = data.Nm
        int [:, ::1] NCable = data.Nc

        double [:, ::1] loadedBCNodes = data.loadedBCNodes
        double [:, ::1] loadedBCEdges = data.loadedBCEdges

        double [:, ::1] Ew = np.zeros((data.nelemsMem, 3), dtype=np.double)

        double [:] V =       np.zeros(data.ndof, dtype=np.double)
        double [:] MV =      np.zeros(data.ndof, dtype=np.double)
        double [:] MinvR =   np.zeros(data.ndof, dtype=np.double)
        double [:] Fint =    np.zeros(data.ndof, dtype=np.double)
        double [:] R =       np.ones(data.ndof, dtype=np.double)
        double [:] RHS =     np.zeros(data.ndof, dtype=np.double)
        double [:] RHS0 =    np.zeros(data.ndof, dtype=np.double)
        double [:] u =       data.u
        double [:] uDot =    np.zeros(data.ndof, dtype=np.double)
        double [:] uDotDot = np.zeros(data.ndof, dtype=np.double)
        double [:] Minv =    np.zeros(data.ndof, dtype=np.double)
        double [:] M =       np.zeros(data.ndof, dtype=np.double)
        double [:] diagK =   np.zeros(data.ndof, dtype=np.double)

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] Z0 = data.Z0
        double [:] X =  data.X
        double [:] Y =  data.Y
        double [:] Z =  data.Z

        long double [:] J11Vec = data.J11Vec
        long double [:] J12Vec = data.J12Vec
        long double [:] J22Vec = data.J22Vec

        double [:] area3 =  data.area3
        double [:] thetaVec = data.thetaVec

        double [:] L0 = data.LCable

        double [:] g = np.asarray(data.load["g"], dtype=np.double)
        double [:] p = data.p

        # distributed load from OpenFOAM
        double [:] Sx = data.Sx
        double [:] Sy = data.Sy
        double [:] Sz = data.Sz
        double [:] pFSI = data.pFSI

        double [:] RF =    np.ones(len(data.dofFixed), dtype=np.double)

        unsigned int [:] row = np.empty(data.nelemsMem * 81 + data.nelemsCable * 36, dtype=np.uintc)
        unsigned int [:] col = np.empty(data.nelemsMem * 81 + data.nelemsCable * 36, dtype=np.uintc)

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

        unsigned int i = 0, iteration = 0, RHS0flag = 0, wrinklingFlag = 0

        int [:] dofFixed = data.dofFixed

        int gravity

        double [:] thickness = data.t
        double [:] area2 = data.area2
        double [:] rho3 = data.rho3
        double [:] rho2 = data.rho2
        double [:] E3 = data.E3
        double [:] E2 = data.E2
        double [:] nu = data.nu

        double alpha = data.props["alpha"]
        double beta =  data.props["beta"]
        double dt =    data.props["dt"]
        double T =     data.props["T"]

        double epsilonKE = data.solverOptions["epsilon_KE"]
        double epsilonR =  data.solverOptions["epsilon_R"]

        double loadStep

        double [:] qq0 = np.zeros(2 * data.nnodes * data.dim, dtype=np.double)

    if data.gravity is False:
        gravity = 0
    elif data.gravity is True:
        gravity = 1

    # initial displacement
    for i in range(ndof):
        qq0[i] = u[i]

    # consistency check for input
    # if sum(rho3) == 0:
    #     raise Exception("No density for membrane given.")
    if dt == 0:
        raise Exception("dt is undefined.")
    if alpha is None:
        alpha = 0
    if beta is None:
        beta = 0
    if T == 0:
        raise Exception("No time defined.")

    damper = data.loadedBCNodes_damper

    # assemble inverse mass matrix as vector format
    assembleM(NMem, NCable, M, Minv, area3, L0, thickness, area2, rho3, rho2, nelemsMem, nelemsCable, dim)

    # TODO: implement into M assembler
    if dim == 2:
        for i in range(np.size(damper,0)):

            # mass matrix
            M[2*(int(damper[i,1])+1)-2] += damper[i,3]
            M[2*(int(damper[i,1])+1)-1] += damper[i,3]

        for i in range(ndof):

            Minv[i] = 1 / M[i]

    elif dim == 3:
        for i in range(np.size(damper,0)):

            # mass matrix
            M[3*(int(damper[i,1])+1)-3] += damper[i,3]
            M[3*(int(damper[i,1])+1)-2] += damper[i,3]
            M[3*(int(damper[i,1])+1)-1] += damper[i,3]

        for i in range(ndof):

            Minv[i] = 1 / M[i]

    # determine sparsity pattern
    orderTemp, indptrTemp, indicesTemp = sparsityPattern(NCable, NMem, nelemsCable, nelemsMem, ndof, dim)

    cdef:
        unsigned int [:] order = np.asarray(orderTemp, dtype=np.uintc)
        unsigned int [:] indptr = np.asarray(indptrTemp, dtype=np.uintc)
        unsigned int [:] indices = np.asarray(indicesTemp, dtype=np.uintc)
        unsigned int [:] state = np.ones(nelemsMem, dtype=np.uintc) * 2
        double [:] dataK = np.zeros(len(indices), dtype=np.double)

        double [:] force_vector = data.force_vector
        double [:] P = np.ones(nelemsMem, dtype=np.double)
        double [:] theta_vec = np.zeros(nelemsMem, dtype=np.double)
        unsigned int wrinkling_iter = 0
        unsigned int iter_goal = data.solverOptions["iter_goal"]
        double sigma_max = data.solverOptions["sigma_max"]

        double [:] pre_stress_cable = data.pre_stress_cable
        double [:] pre_strain_cable = data.pre_strain_cable

        double [:, ::1] pre_stress_membrane = data.pre_stress_membrane
        double [:, ::1] pre_strain_membrane = data.pre_strain_membrane

        unsigned int [:] pre_active = data.pre_active

    wrinkling_model = data.solverOptions["wrinkling_model"]
    wrinkling = 1



    # ODE solver
    from scipy.integrate import ode

    integrator = ode(firstOrderSystemNonlinear)

    integrator.set_integrator("vode", method='bdf', order=15, nsteps=3000)#, rtol=1E-16, nsteps=50000)

    integrator.set_initial_value(qq0, 0).set_f_params(X, Y, Z, X0, Y0, Z0, Minv, Fint, RHS, RHS0, uDot, uDotDot,
                                                     NMem, NCable, area3, L0, elPressurised, elFSI, p, Sx, Sy, Sz,
                                                     pFSI, loadedBCNodes, loadedBCEdges, dofFixed, dim, nnodes, ndof,
                                                     gravity, nelemsMem, nelemsCable, rho3, area2, rho2,
                                                     J11Vec, J22Vec,J12Vec, E3, E2, nu, thickness, alpha, beta,
                                                     thetaVec, Ew, order, indices, indptr, dataK, diagK, wrinkling,
                                                     g, i, wrinklingFlag, nPressurised, nFSI, state, force_vector,
                                                     P, theta_vec, wrinkling_iter, iter_goal, sigma_max,
                                                     pre_stress_cable, pre_strain_cable, pre_stress_membrane,
                                                     pre_strain_membrane, pre_active, damper)

    counter = 0
    data.time = 0

    time_array = np.linspace(0., T, int(T/dt + 1))

    # from scipy.integrate import odeint
    # solution = odeint(firstOrderSystemNonlinear, qq0, time_array, args=(X, Y, Z, X0, Y0, Z0, Minv, Fint, RHS, RHS0, uDot, uDotDot,
    #                                                                    NMem, NCable, area3, L0, elPressurised, elFSI, p, Sx, Sy, Sz,
    #                                                                    pFSI, loadedBCNodes, loadedBCEdges, dofFixed, dim, nnodes, ndof,
    #                                                                    gravity, nelemsMem, nelemsCable, rho3, area2, rho2,
    #                                                                    J11Vec, J22Vec,J12Vec, E3, E2, nu, thickness, alpha, beta,
    #                                                                    thetaVec, Ew, order, indices, indptr, dataK, diagK, wrinkling,
    #                                                                    g, i, wrinklingFlag, nPressurised, nFSI, state, force_vector,
    #                                                                    P, theta_vec, wrinkling_iter, iter_goal, sigma_max,
    #                                                                    pre_stress_cable, pre_strain_cable, pre_stress_membrane,
    #                                                                    pre_strain_membrane, pre_active, damper))

    # # write output files
    # for time_step in range(np.size(solution, 1)):
    #
    #     if time_step % data.writeInterval == 0:
    #
    #         u_vec = solution[time_step, 0:ndof]
    #         v_vec = solution[time_step, ndof::]
    #
    #         # update current configuration
    #         if dim == 2:
    #             for i in range(nnodes):
    #                 X[i] = X0[i] + u_vec[2 * (i + 1) - 2]
    #                 Y[i] = Y0[i] + u_vec[2 * (i + 1) - 1]
    #         elif dim == 3:
    #             for i in range(nnodes):
    #                 X[i] = X0[i] + u_vec[3 * (i + 1) - 3]
    #                 Y[i] = Y0[i] + u_vec[3 * (i + 1) - 2]
    #                 Z[i] = Z0[i] + u_vec[3 * (i + 1) - 1]
    #
    #         # RHS vector in current configuration
    #         assembleRHS(X, Y, Z, u_vec, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0, gravity,
    #                     nelemsMem, nelemsCable, nPressurised, nFSI, thickness, rho3, area2, rho2, g, Sx, Sy, Sz,
    #                     pFSI, loadedBCNodes, loadedBCEdges, RHS0flag, 1, dim, force_vector, E2,
    #                     pre_stress_cable, pre_strain_cable, pre_stress_membrane, pre_strain_membrane,
    #                     pre_active, J11Vec, J22Vec, J12Vec, thetaVec, E3, nu)
    #
    #         computeuDotDot(Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec, nelemsMem, nelemsCable, ndof, E3, E2,
    #                        nu, thickness, area2, alpha, beta, thetaVec, area3, L0, Fint, RHS, Ew, p, order, indices,
    #                        indptr, state, dataK, uDot, uDotDot, diagK, dim, wrinklingFlag, P, theta_vec, wrinkling_iter,
    #                        iter_goal, sigma_max, damper)
    #
    #         # Update data object
    #         data.X = X
    #         data.Y = Y
    #         data.Z = Z
    #         data.u = u_vec
    #         data.V = v_vec
    #         data.RHS = RHS
    #         data.Ew = Ew
    #         data.RF = RF
    #         data.Fint = Fint
    #         data.R = R
    #         data.M = M
    #
    #         data.P = P
    #         data.theta_vec = theta_vec
    #
    #         data.time = time_step
    #
    #         # Write output into vtk file
    #         if data.autoWrite is True:
    #             writeVTK(data)
    #
    #
    # import matplotlib.pyplot as plt
    # plt.subplot(2,1,1)
    # plt.plot(time_array, solution[:,ndof-2])
    # plt.grid()
    # plt.ylabel('Position [m]')
    #
    # plt.subplot(2,1,2)
    # plt.plot(time_array, solution[:,2*ndof-1])
    # plt.grid()
    # plt.ylabel('Velocity [m/s]')
    # plt.xlabel('Time [s]')
    #
    # plt.savefig('pos_and_vel_vs_time.png')



    for time_ind in range(1, len(time_array)):

        qq = integrator.integrate(time_array[time_ind])

        assert integrator.successful()

        data.kinetic_energy.append(qq[ndof-38])
        data.strain_energy.append(qq[2*ndof-38])

        if counter % data.writeInterval == 0:

            if data.silent is False:
                print("current time = {}".format(integrator.t))

            q1, q2 = np.reshape(qq, (2, -1))

            # update current configuration
            if dim == 2:
                for i in range(nnodes):
                    X[i] = X0[i] + q1[2 * (i + 1) - 2]
                    Y[i] = Y0[i] + q1[2 * (i + 1) - 1]
            elif dim == 3:
                for i in range(nnodes):
                    X[i] = X0[i] + q1[3 * (i + 1) - 3]
                    Y[i] = Y0[i] + q1[3 * (i + 1) - 2]
                    Z[i] = Z0[i] + q1[3 * (i + 1) - 1]

            # Update data object
            data.X = X
            data.Y = Y
            data.Z = Z
            data.u = q1
            data.V = q2
            data.RHS = RHS
            data.Ew = Ew
            data.RF = RF
            data.Fint = Fint
            data.R = R
            data.M = M

            data.P = P
            data.theta_vec = theta_vec

            data.time = counter

            # Write output into vtk file
            if data.autoWrite is True:
                writeVTK(data)

        counter += 1

    data.time_array = time_array

    # reset displacements
    u[...] = 0.


def firstOrderSystemNonlinear(t, qq, X, Y, Z, X0, Y0, Z0, Minv, Fint, RHS, RHS0, uDot, uDotDot, NMem, NCable, area3,
                              L0, elPressurised, elFSI, p, Sx, Sy, Sz, pFSI, loadedBCNodes, loadedBCEdges, dofFixed,
                              dim, nnodes, ndof, gravity, nelemsMem, nelemsCable, rho3, area2, rho2, J11Vec, J22Vec,
                              J12Vec, E3, E2, nu, thickness, alpha, beta, thetaVec, Ew, order, indices, indptr,
                              data, diagK, wrinkling, g, i, wrinklingFlag, nPressurised, nFSI, state, force_vector,
                              P, theta_vec, wrinkling_iter, iter_goal, sigma_max, pre_stress_cable, pre_strain_cable,
                              pre_stress_membrane, pre_strain_membrane, pre_active, damper):
        """
        converts M a + C v + Fint(u) = F(u) into first order system for scipy solver
        and computes nodal acceleration uDotDot = Minv * (RHS - Fint) - alpha * uDot - beta * Minv * K * uDot
        :param t: time vector
        :param w: state vector (q1 = u, q2 = v)
        :param RHS: external load vector
        :return: state space matrix
        """
        # cdef unsigned int RHS0flag, i, wrinklingFlag

        # update current configuration
        if dim == 2:
            for i in range(nnodes):
                X[i] = X0[i] + qq[2 * (i + 1) - 2]
                Y[i] = Y0[i] + qq[2 * (i + 1) - 1]
        elif dim == 3:
            for i in range(nnodes):
                X[i] = X0[i] + qq[3 * (i + 1) - 3]
                Y[i] = Y0[i] + qq[3 * (i + 1) - 2]
                Z[i] = Z0[i] + qq[3 * (i + 1) - 1]

        if t == 0:
            RHS0flag = 0
            wrinklingFlag = 0
        else:
            RHS0flag = 1
            if wrinkling is True:
                wrinklingFlag = 1
            else:
                wrinklingFlag = 0

        for i in range(ndof):
            uDot[i] = qq[i + ndof]

        q1, q2 = np.reshape(qq, (2, -1))

        if t == 0:
            pre_u = q1
        else:
            pre_u = np.zeros(len(q1))

        # RHS vector in current configuration
        assembleRHS(X, Y, Z, pre_u, NMem, NCable, p, RHS, RHS0, elPressurised, elFSI, area3, L0, gravity,
                    nelemsMem, nelemsCable, nPressurised, nFSI, thickness, rho3, area2, rho2, g, Sx, Sy, Sz,
                    pFSI, loadedBCNodes, loadedBCEdges, RHS0flag, 1, dim, force_vector, E2,
                    pre_stress_cable, pre_strain_cable, pre_stress_membrane, pre_strain_membrane,
                    pre_active, J11Vec, J22Vec, J12Vec, thetaVec, E3, nu)

        if t == 0:
            for i in range(len(pre_active)):
                pre_active[i] = 0

        # Ew[...] = 0

        computeuDotDot(Minv, X, Y, Z, NMem, NCable, J11Vec, J22Vec, J12Vec, nelemsMem, nelemsCable, ndof, E3, E2,
                       nu, thickness, area2, alpha, beta, thetaVec, area3, L0, Fint, RHS, Ew, p, order, indices,
                       indptr, state, data, uDot, uDotDot, diagK, dim, wrinklingFlag, P, theta_vec, wrinkling_iter,
                       iter_goal, sigma_max, damper)

        wrinkling_iter += 1

        # set Dirichlet nodes to zero
        correctBC(uDotDot, dofFixed)
        correctBC(uDot, dofFixed)

        # TODO: REMOVE
        uDotDot[0] = 0
        uDotDot[1] = 0
        uDot[0] = 0
        uDot[1] = 0

        qq_return = []
        # update states
        for i in range(ndof):
            qq_return.append(uDot[i])
        for i in range(ndof):
            qq_return.append(uDotDot[i])
            # qq[i + ndof] = uDotDot[i]

        return qq_return
