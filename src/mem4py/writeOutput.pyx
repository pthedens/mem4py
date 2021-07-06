# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
import os
cimport numpy as np
cimport cython
from datetime import datetime

from mem4py.helper.computeStress cimport computeStress
from mem4py.helper.writeObject cimport writeVectorNode3D
from mem4py.helper.writeObject cimport writeVectorNode2D
from mem4py.helper.writeObject cimport writeVectorElement
from mem4py.helper.writeObject cimport writeScalarElement
from mem4py.helper.writeObject cimport writeScalarElementInt
from mem4py.helper.normalVector cimport computeNormalVector


cdef extern from "math.h":
    double sqrt(double m)
    double atan2(double m, double n)


cdef int writeVTK(object data) except -1:

    cdef:

        str inputFile = data.inputName

        unsigned int nnodes = data.nnodes
        unsigned int nelems = data.nelems
        unsigned int nelemsMem = data.nelemsMem
        unsigned int nelemsCable = data.nelemsCable
        unsigned int ndof = data.ndof
        unsigned int dim = data.dim
        unsigned int wrinkle = 0
        Py_ssize_t n, el, index, i

        int time = data.time
        int tLength = data.tLength

        double [:] t = data.t
        double [:] area2 = data.area2
        double [:] rho3 = data.rho3
        double [:] rho2 = data.rho2
        double [:] E_mod_3 = data.E3
        double [:] E_mod_2 = data.E2
        double [:] nu = data.nu

        int [:] dofFixed = data.dofFixed

        unsigned int [:] elPressurised = data.elPressurised
        unsigned int nPressurised = data.nPressurised

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] Z0 = data.Z0

        double [:] u = data.u
        double [:] V = data.V
        double [:] acc = data.acc

        double [:] X = data.X
        double [:] Y = data.Y
        double [:] Z = data.Z

        double [:] RHS = data.RHS
        double [:] Fint = data.Fint
        double [:] R = data.R
        double [:] p = np.zeros(data.nelems, dtype=np.double)
        double [:] M = data.M

        long double [:] J11Vec = data.J11Vec
        long double [:] J12Vec = data.J12Vec
        long double [:] J22Vec = data.J22Vec

        double [:] thetaVec = data.thetaVec
        double [:] area3 = data.area3
        double [:] L0 = data.LCable

        double [:] dofFixedWrite = np.zeros(ndof, dtype=np.double)

        double [:] VMS = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] S1 = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] S2 = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] sigma_cable = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] force_cable = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] E1 = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] E2 = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] eps_cable = np.zeros(nelemsMem + nelemsCable, dtype=np.double)

        double [:] SE_cable = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] KE_cable = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] SE_membrane = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] KE_membrane = np.zeros(nelemsMem + nelemsCable, dtype=np.double)

        double [:] RF = np.zeros(ndof, dtype=np.double)

        double [:, ::1] S = np.zeros((nelemsMem + nelemsCable, 3), dtype=np.double)
        double [:, ::1] cauchy = np.zeros((nelemsMem + nelemsCable, 3), dtype=np.double)
        double [:, ::1] E = np.zeros((nelemsMem + nelemsCable, 3), dtype=np.double)
        double [:, ::1] Ew = np.zeros((nelemsMem + nelemsCable, 3), dtype=np.double)
        double [:, ::1] normalVector = np.zeros((nelemsMem + nelemsCable, 3), dtype=np.double)

        int [:, ::1] NMem = data.Nm
        int [:, ::1] NCable = data.Nc

        unsigned int [:] state = np.zeros(nelemsMem + nelemsCable, dtype=np.uintc)
        unsigned int [:] state_cable = np.zeros(nelemsCable, dtype=np.uintc)
        unsigned int [:] state_mem = np.zeros(nelemsMem, dtype=np.uintc)

        double [:] pre_stress_cable = np.zeros(nelemsMem + nelemsCable, dtype=np.double)
        double [:] pre_strain_cable = np.zeros(nelemsMem + nelemsCable, dtype=np.double)

        double [:, ::1] pre_stress_membrane = np.zeros((nelemsMem + nelemsCable, 3), dtype=np.double)
        double [:, ::1] pre_strain_membrane = np.zeros((nelemsMem + nelemsCable, 3), dtype=np.double)

    wrinkling_model = data.solverOptions["wrinkling_model"]

    if data.Sw:
        print("writing Sw instead of S")
        length_Sw = len(data.Sw)
        S = np.asarray(data.Sw)
    else:
        length_Sw = 0

    # for i in range(len(state)):
    #     state[i + nelemsCable] = 2

    for el in dofFixed:
        dofFixedWrite[el] = 1

    if nPressurised > 0:
        for el in range(nPressurised):
            p[nelemsCable + elPressurised[el]] = data.p[el]

    # reaction forces
    index = 0
    for el in dofFixed:
        RF[el] = data.RF[index]
        index += 1

    for el in range(nelemsMem):
        Ew[nelemsCable + el, 0] = data.Ew[el, 0]
        Ew[nelemsCable + el, 1] = data.Ew[el, 1]
        Ew[nelemsCable + el, 2] = data.Ew[el, 2]

    if data.pre_active[0] == 1:

        for el in range(nelemsCable):

            pre_stress_cable[el] = data.pre_stress_cable[el]

    if data.pre_active[1] == 1:

        for el in range(nelemsCable):

            pre_strain_cable[el] = data.pre_strain_cable[el]

    if data.pre_active[2] == 1:

        for el in range(nelemsMem):

            pre_stress_membrane[el + nelemsCable, 0] = data.pre_stress_membrane[el, 0]
            pre_stress_membrane[el + nelemsCable, 1] = data.pre_stress_membrane[el, 1]
            pre_stress_membrane[el + nelemsCable, 2] = data.pre_stress_membrane[el, 2]

    if data.pre_active[3] == 1:

        for el in range(nelemsMem):

            pre_strain_membrane[el + nelemsCable, 0] = data.pre_strain_membrane[el, 0]
            pre_strain_membrane[el + nelemsCable, 1] = data.pre_strain_membrane[el, 1]
            pre_strain_membrane[el + nelemsCable, 2] = data.pre_strain_membrane[el, 2]

    # compute Cauchy stress and strain
    computeStress(X, Y, Z, NMem, NCable, nelemsCable, nelemsMem, J11Vec, J12Vec,
                  J22Vec, E_mod_3, nu, E_mod_2, area2, S, cauchy, sigma_cable, S1, S2, E1, E2, E,
                  eps_cable, thetaVec, VMS, area3, L0, Ew, state_cable, state_mem, wrinkle,
                  dim, data.P, data.alpha_array, data.solverOptions["sigma_max"], wrinkling_model)

    # forces in cable
    if nelemsCable > 0:
        for n in range(nelemsCable):
            force_cable[n] = sigma_cable[n] * area2[n]

    # energy
    # unsigned int [:] allDofCable = np.empty(2 * dim, dtype=np.uintc)
    # unsigned int [:] allDofMem = np.empty(3 * dim, dtype=np.uintc)
    index = 0
    for el in range(nelemsCable):

        SE_cable[index] = 0.5 * sigma_cable[el] * eps_cable[el] * area2[el] * L0[el]

        index += 1

        # for i in range(dim):
        #     allDofCable[i] = dim * (NCable[el, 1] + 1) - dim + i
        #     allDofCable[i+dim] = dim * (NCable[el, 2] + 1) - dim + i
        #
        # KE_cable[index] = 0.5 * rho2[el] * area2[el] * L0[el] *

    for el in range(nelemsMem):

        SE_membrane[index] = 0.5 * area3[el] * t[el] * (S[el, 0] * E[el, 0] +
                                                        S[el, 1] * E[el, 1] +
                                                        S[el, 2] * E[el, 2])

        index += 1

    # element states
    index = 0
    for el in range(nelemsCable):
        state[index] = state_cable[el]
        index += 1
    for el in range(nelemsMem):
        state[index] = state_mem[el]
        index += 1

    # save field properties in class object mem4py as numpy array
    # TODO this causes a bug when analysis is restarted...
    # data.X0 = np.asarray(X0)
    # data.Y0 = np.asarray(Y0)
    # data.Z0 = np.asarray(Z0)
    # data.X = np.asarray(X)
    # data.Y = np.asarray(Y)
    # data.Z = np.asarray(Z)
    # data.E = E
    # data.E1 = np.asarray(E1)
    # data.E2 = np.asarray(E2)
    # data.eps_cable = np.asarray(eps_cable)
    # data.Ew = np.asarray(Ew)
    # data.S = S
    # data.S1 = np.asarray(S1)
    # data.S2 = np.asarray(S2)
    # data.sigma_cable = np.asarray(sigma_cable)
    # data.VMS = np.asarray(VMS)
    # data.state = np.asarray(state)
    # data.p = np.asarray(p)
    cdef double a,b
    cdef double [:] PK2_S1 = np.zeros(nelemsMem + nelemsCable, dtype=np.double)

    # TODO: causes out of bounds error
    # for el in range(len(PK2_S1)):
    #     a = (S[nelemsCable + el, 0] + S[nelemsCable + el, 1]) / 2
    #     b = S[nelemsCable + el, 2] * S[nelemsCable + el, 2] - \
    #         S[nelemsCable + el, 0] * S[nelemsCable + el, 1]
    #     # sigma1
    #     PK2_S1[nelemsCable + el] = a + sqrt(a * a + b)
    #
    data.S1_output = np.copy(np.asarray(S1))
    # data.E_output = np.copy(np.asarray(E))
    # data.S_output = np.copy(np.asarray(S))

    # compute area normal vectors of pressurized elements
    if dim == 3:
        computeNormalVector(NMem, X, Y, Z, elPressurised, normalVector, nelemsCable)

    # if time == 0:
    #     outputDirectory = "output_mem4py/" + datetime.now().strftime("%Y%m%d-%H%M") + "/"
    #     data.output_dir = outputDirectory
    outputDirectory = "output_mem4py"


    if not os.path.exists(outputDirectory):
           os.makedirs(outputDirectory)

    if time or time == 0:
        name = "{}/{}_{:0>{}}.vtk".format(outputDirectory, inputFile, time, tLength)
    else:
        name = "{}/{}.vtk".format(outputDirectory,inputFile)

    with open(name, 'w') as fout:

            fout.write('# vtk DataFile Version 2.0\nmem4py VTK writer\nASCII\nDATASET '
                       'UNSTRUCTURED_GRID\nPOINTS %s float\n' % nnodes)

            # Write undeformed node coordinates
            for n in range(0, nnodes):
                fout.write('%s %s %s\n' % (X0[n], Y0[n], Z0[n]))  # initial (X,Y,Z) nodal coordinates

            # Write element connectivity (3 for triangle, 2 for cable)
            fout.write('\nCELLS %s %s \n' % (nelems, (nelemsMem * 4 + nelemsCable * 3)))

            if nelemsCable != 0:
                for el in range(nelemsCable):
                    fout.write(
                        '2 %s %s\n' % (NCable[el, 1], NCable[el, 2]))  # connectivity for cable
            if nelemsMem != 0:
                for el in range(nelemsMem):
                    fout.write(
                        '3 %s %s %s\n' % (NMem[el, 1], NMem[el, 2], NMem[el, 3]))  # connectivity for triangle

            # Write cell types
            fout.write('\nCELL_TYPES %s \n' % nelems)  # 3 for triangle (3 nodes)
            for n in range(nelemsCable):
                fout.write('3\n')  # 3 = line
            for n in range(nelemsMem):
                fout.write('5\n')  # 5 = triangle

            # Start point data (e.g. displacements)
            fout.write('\nPOINT_DATA %s \n' % nnodes)

            if dim == 2:

                # Write displacements
                writeVectorNode2D(u, nnodes, 'u', fout)

                # Write velocities
                writeVectorNode2D(V, nnodes, 'V', fout)

                # Write accelerations
                writeVectorNode2D(acc, nnodes, 'acc', fout)

                # Write fixed nodes
                writeVectorNode2D(dofFixedWrite, nnodes, 'fix', fout)

                # Write fixed nodes
                writeVectorNode2D(RF, nnodes, 'F_reaction', fout)

                # Write external load vector (right hand side)
                writeVectorNode2D(RHS, nnodes, 'F_external', fout)

                # # Write internal load vector (left hand side)
                # writeVectorNode2D(Fint, nnodes, 'F_internal', fout)

            elif dim == 3:

                # Write displacements
                writeVectorNode3D(u, nnodes, 'u', fout)

                # Write velocities
                writeVectorNode3D(V, nnodes, 'V', fout)

                # Write accelerations
                writeVectorNode3D(acc, nnodes, 'acc', fout)

                # Write fixed nodes
                writeVectorNode3D(dofFixedWrite, nnodes, 'fix', fout)

                # Write fixed nodes
                writeVectorNode3D(RF, nnodes, 'F_reaction', fout)

                # Write external load vector (right hand side)
                writeVectorNode3D(RHS, nnodes, 'F_external', fout)

                # Write residual vector (RHS - LHS)
                writeVectorNode3D(R, nnodes, 'F_residual', fout)

                # dof mass
                writeVectorNode3D(M, nnodes, 'M', fout)

            # Start cell data (e.g. strain and stress)
            fout.write('\nCELL_DATA %s \n' % (nelemsCable + nelemsMem))

            # cable elements
            if nelemsCable > 0:

                # Green's strain in cable
                writeScalarElement(eps_cable, 'Cable_strain', nelemsCable + nelemsMem, fout)

                # Cauchy stress in cable
                writeScalarElement(sigma_cable, 'Cable_stress', nelemsCable + nelemsMem, fout)

                # Force in cable
                writeScalarElement(force_cable, 'Cable_force', nelemsCable + nelemsMem, fout)

                # Strain energy in cable
                writeScalarElement(SE_cable, 'Cable_SE', nelemsCable + nelemsMem, fout)

                # Kinetic energy in cable
                # writeScalarElement(KE_cable, 'Cable_KE', nelemsCable + nelemsMem, fout)

                if data.pre_active[0] == 1:

                    writeScalarElement(pre_stress_cable, "Cable_pre_S", nelemsCable + nelemsMem, fout)

                if data.pre_active[1] == 1:

                    writeScalarElement(pre_strain_cable, "Cable_pre_E", nelemsCable + nelemsMem, fout)

            # membrane elements
            if nelemsMem > 0:

                # Green's strain tensor
                writeVectorElement(E, 'Membrane_E', nelemsCable + nelemsMem, fout)

                # principal strains
                writeScalarElement(E1, 'Membrane_E1', nelemsCable + nelemsMem, fout)
                writeScalarElement(E2, 'Membrane_E2', nelemsCable + nelemsMem, fout)

                # Cauchy stress tensor
                writeVectorElement(S, 'Membrane_PK2', nelemsCable + nelemsMem, fout)
                writeVectorElement(cauchy, 'Membrane_Cauchy', nelemsCable + nelemsMem, fout)

                # principal stress
                writeScalarElement(S1, 'Membrane_S1', nelemsCable + nelemsMem, fout)
                writeScalarElement(S2, 'Membrane_S2', nelemsCable + nelemsMem, fout)

                # Von Mises stress
                writeScalarElement(VMS, "Membrane_VMS", nelemsCable + nelemsMem, fout)

                # Strain energy in cable
                writeScalarElement(SE_membrane, 'Membrane_SE', nelemsCable + nelemsMem, fout)

                # Kinetic energy in cable
                # writeScalarElement(KE_membrane, 'Membrane_KE', nelemsCable + nelemsMem, fout)

                if np.sum(NMem[:, 4]) > 0:

                    # wrinkling strain Ew
                    # writeVectorElement(Ew, 'Ew', nelemsCable + nelemsMem, fout)

                    # Write membrane state (0=slack, 1=wrinkle, 2=taut)
                    writeScalarElementInt(state, 'Membrane_state', nelemsCable + nelemsMem, fout)

                if nPressurised > 0:

                    # write normal vectors of pressurised elements
                    writeVectorElement(normalVector, 'n', nelemsCable + nelemsMem, fout)

                    writeScalarElement(p, "Membrane_pressure", nelemsCable + nelemsMem, fout)

                if data.pre_active[2] == 1:

                    writeVectorElement(pre_stress_membrane, "Membrane_pre_S", nelemsCable + nelemsMem, fout)

                if data.pre_active[3] == 1:

                    writeVectorElement(pre_strain_membrane, "Membrane_pre_E", nelemsCable + nelemsMem, fout)

    if data.silent is False:
        print("Finished writing output to {}".format(name))
