import numpy as np
cimport numpy as np
cimport cython

from src.assembler.M3D cimport M3DExplicit
from src.writeOutput cimport writeVTK3D
from src.assembler.RHS cimport RHS3D
from src.helper.dirichletHandler cimport correctBC3D
from src.assembler.M3D cimport K3DIsotropic

from src.ceygen.math cimport subtract_vv

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int solve3DDynamicExplicit(object data) except -1:

    # type def variables
    cdef:

        int [:, ::1] NMem = data.Nm
        int [:, ::1] NBar = data.Nb

        unsigned int [:, ::1] loadedBCNodes = data.loadedBCNodes
        unsigned int [:, ::1] loadedBCEdges = data.loadedBCEdges

        double [:, ::1] Ew = np.zeros((data.nelemsMem, 3), dtype=np.double)

        double [:] V =     np.zeros(data.ndof, dtype=np.double)
        double [:] MV =    np.zeros(data.ndof, dtype=np.double)
        double [:] MinvR = np.zeros(data.ndof, dtype=np.double)
        double [:] Fint =  np.zeros(data.ndof, dtype=np.double)
        double [:] R =     np.ones(data.ndof, dtype=np.double)
        double [:] RHS =   np.zeros(data.ndof, dtype=np.double)
        double [:] RHS0 = np.zeros(data.ndof, dtype=np.double)
        double [:] u =     np.zeros(data.ndof, dtype=np.double)
        double [:] Minv =  np.zeros(data.ndof, dtype=np.double)
        double [:] M =     np.zeros(data.ndof, dtype=np.double)

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] Z0 = data.Z0
        double [:] X =  data.X
        double [:] Y =  data.Y
        double [:] Z =  data.Z

        double [:] J11Vec = data.J11Vec
        double [:] J12Vec = data.J12Vec
        double [:] J22Vec = data.J22Vec

        double [:] areaVec =  data.areaVec
        double [:] thetaVec = data.thetaVec

        double [:] L0 = data.LBar

        double [:] g = np.array([data.load["gX"], data.load["gY"], data.load["gZ"]], dtype=np.double)
        double [:] p = data.p

        # distributed load from OpenFOAM
        double [:] Sx = data.Sx
        double [:] Sy = data.Sy
        double [:] Sz = data.Sz

        double [:] RF =    np.ones(len(data.dofFixed), dtype=np.double)

        unsigned int [:] row = np.empty(data.nelemsMem * 81 + data.nelemsBar * 36, dtype=np.uintc)
        unsigned int [:] col = np.empty(data.nelemsMem * 81 + data.nelemsBar * 36, dtype=np.uintc)

        unsigned int [:] elPressurised = data.elPressurised

        unsigned int nelemsMem = data.nelemsMem
        unsigned int nelemsBar = data.nelemsBar
        unsigned int nnodes =    data.nnodes
        unsigned int ndof =      data.ndof

        unsigned int nLoadSteps = data.solverOptions["nLoadSteps"]

        unsigned int i, iteration = 0, RHS0flag = 0

        int [:] dofFixed = data.dofFixed

        int gravity

        object load = data.load

        double t =       data.props["t"]
        double areaBar = data.props["areaBar"]
        double rhoMem =  data.props["rhoMem"]
        double rhoBar =  data.props["rhoBar"]
        double EMem =    data.props["EMem"]
        double EBar =    data.props["EBar"]
        double poisson = data.props["nu"]

        double alpha = data.props["alpha"]
        double beta =  data.props["beta"]
        double dt =    data.props["dt"]
        double T =     data.props["T"]

        double epsilonKE = data.solverOptions["epsilonKE"]
        double epsilonR =  data.solverOptions["epsilonR"]

        double loadStep

    if data.gravity is False:
        gravity = 0
    elif data.gravity is True:
        gravity = 1


    # consistency check for input
    if rhoMem == 0:
        raise Exception("No density for membrane given.")
    if dt == 0:
        raise Exception("dt is undefined.")
    if alpha is None:
        raise Exception("No alpha defined.")
    if beta is None:
        raise Exception("No beta defined.")
    if T == 0:
        raise Exception("No time defined.")

    # assemble inverse mass matrix as vector format
    M3DExplicit(NMem, NBar, Minv, areaVec, L0, t, areaBar, rhoMem, rhoBar, nelemsMem, nelemsBar)

    # solution vector
    w0 = np.hstack([np.zeros(len(u)), np.zeros(len(u))])

    # ODE solver
    from scipy.integrate import ode
    integrator = ode(firstOrderSystemNonlinear)

    integrator.set_integrator("vode", method='bdf', order=5, rtol=1E-16, nsteps=5000)
    integrator.set_initial_value(w0, 0).set_f_params(Minv, RHS, RHS0, alpha, beta, J11Vec, J22Vec, J12Vec, X, Y, Z,
                              NMem, NBar, Fint, EMem, poisson, nelemsMem, nelemsBar, t, areaVec,
                              EBar, areaBar, L0, Sx, Sy, Sz, X0, Y0, Z0, nnodes, dofFixed, p, elPressurised,
                              gravity, ndof, g, Ew, rhoMem, rhoBar, loadedBCNodes,
                              loadedBCEdges, load)

    cdef double [:] qq = np.zeros(2 * data.nnodes * data.dim, dtype=np.double)

    print("starting ode solver")

    while integrator.successful() and integrator.t < T:

        qq = integrator.integrate(integrator.t + dt)
        print("current time = {}".format(integrator.t))

    q1, q2 = np.reshape(qq, (2, -1))
    u = q1

    # update current configuration
    for i in range(nnodes):

        X[i] = X0[i] + u[3 * (i + 1) - 3]
        Y[i] = Y0[i] + u[3 * (i + 1) - 2]
        Z[i] = Z0[i] + u[3 * (i + 1) - 1]

    data.X = X
    data.Y = Y
    data.Z = Z
    data.u = u
    data.RHS = RHS
    data.Ew = Ew
    data.RF = RF

    # Write output into vtk file
    if data.autoWrite is True:
        writeVTK3D(data)


def firstOrderSystemNonlinear(t, w, Minv, RHS, RHS0, alpha, beta, J11Vec, J22Vec, J12Vec, X, Y, Z,
                              NMem, NBar, Fint, EMem, poisson, nelemsMem, nelemsBar, thickness, areaVec,
                              EBar, areaBar, L0, Sx, Sy, Sz, X0, Y0, Z0, nnodes, dofFixed, p, elPressurised,
                              gravity, ndof, g, Ew, rhoMem, rhoBar, loadedBCNodes,
                              loadedBCEdges, load):
        """
        converts M a + C v + K(u) u = F(u) into first order system for scipy solver
        :param t: time vector
        :param w: state vector (q1 = u, q2 = v)
        :param RHS: external load vector
        :return: state space matrix
        """

        cdef:

            double [:] R = np.empty(ndof, dtype=np.double)

        q1, q2 = np.reshape(w, (2, -1))

        for i in range(nnodes):

            X[i] = X0[i] + q1[3 * (i + 1) - 3]
            Y[i] = Y0[i] + q1[3 * (i + 1) - 2]
            Z[i] = Z0[i] + q1[3 * (i + 1) - 1]

        # RHS vector in current configuration
        RHS3D(X, Y, Z, NMem, NBar, p, RHS, RHS0, elPressurised, areaVec, L0, gravity, nelemsMem,
              nelemsBar, thickness, rhoMem, areaBar, rhoBar, g, Sx, Sy, Sz, loadedBCNodes, loadedBCEdges,
              0, load, 1)

        K = K3DIsotropic(X, Y, Z, NMem, NBar, J11Vec, J22Vec, J12Vec, nelemsMem, nelemsBar, ndof,
                             EMem, poisson, thickness, areaVec, EBar, areaBar, L0, Fint, p)

        # subtract_vv(RHS, Fint, R)
        R = np.asarray(RHS) - np.asarray(Fint)

        # BCs
        correctBC3D(R, dofFixed)

        udd = np.asarray(np.multiply(Minv, R)) - alpha * q2 - \
              beta * np.multiply(Minv, np.asarray(K.multiply(q2).sum(axis=1))[:, 0])

        correctBC3D(udd, dofFixed)

        q = np.append(q2, udd)

        return q
