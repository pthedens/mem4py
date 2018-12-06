import numpy as np
import os
cimport numpy as np
cimport cython

from src.helper.computeStress cimport computeStress3D
from src.helper.computeStress cimport computeStress2D

from src.helper.writeObject cimport writeVectorNode3D
from src.helper.writeObject cimport writeVectorNode2D
from src.helper.writeObject cimport writeVectorElement
from src.helper.writeObject cimport writeScalarElement
from src.helper.writeObject cimport writeScalarElementInt


cdef extern from "math.h":
    double sqrt(double m)
    double atan2(double m, double n)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int writeVTK2D(object data) except -1:

    cdef:

        str inputFile = data.inputName

        unsigned int nnodes = data.nnodes
        unsigned int nelems = data.nelems
        unsigned int nelemsMem = data.nelemsMem
        unsigned int nelemsBar = data.nelemsBar
        unsigned int ndof = data.ndof
        unsigned int n, el, index

        int time = data.time

        double t =       data.props["t"]
        double areaBar = data.props["areaBar"]
        double rhoMem =  data.props["rhoMem"]
        double rhoBar =  data.props["rhoBar"]
        double EMem =    data.props["EMem"]
        double EBar =    data.props["EBar"]
        double poisson = data.props["nu"]

        int [:] dofFixed = data.dofFixed

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] u = data.u
        double [:] X = data.X
        double [:] Y = data.Y
        double [:] J11Vec = data.J11Vec
        double [:] J12Vec = data.J12Vec
        double [:] J22Vec = data.J22Vec
        double [:] thetaVec = data.thetaVec
        double [:] areaVec = data.areaVec
        double [:] L0 = data.LBar
        double [:] dofFixedWrite = np.ones(ndof, dtype=np.double)
        double [:] VMS = np.zeros(nelemsMem + nelemsBar, dtype=np.double)
        double [:] RF = np.zeros(ndof, dtype=np.double)

        double [:, ::1] S = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)
        double [:, ::1] Sp = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)
        double [:, ::1] Ep = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)
        double [:, ::1] Eelastic = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)
        double [:, ::1] Ew = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)

        int [:, ::1] NMem = data.Nm
        int [:, ::1] NBar = data.Nb

        unsigned int [:] state = np.zeros(nelemsMem + nelemsBar, dtype=np.uintc)

    for el in dofFixed:
        dofFixedWrite[el] = 0

    # reaction forces
    index = 0
    for el in dofFixed:
        RF[el] = data.RF[index]
        index += 1
    
    # compute Cauchy stress and strain
    computeStress2D(X, Y, NMem, NBar, nelemsBar, nelemsMem, J11Vec, J12Vec, J22Vec, EMem,
                    poisson, EBar, areaBar, S, Sp, Ep, Eelastic, thetaVec, VMS, areaVec, L0,
                    Ew, state)
    
    outputDirectory = "output4mem"

    if not os.path.exists(outputDirectory):
           os.makedirs(outputDirectory)
    if time:
        name = "{}/{}_{:0>{}}.vtk".format(outputDirectory, inputFile, time, data.tLength)
    else:
        name = "{}/{}.vtk".format(outputDirectory,inputFile)

    with open(name, 'w') as fout:

            fout.write('# vtk DataFile Version 2.0\nmem4py VTK writer\nASCII\nDATASET '
                       'UNSTRUCTURED_GRID\nPOINTS %s float\n' % nnodes)

            # Write undeformed node coordinates
            for n in range(0, nnodes):
                fout.write('%s %s %s\n' % (X0[n], Y0[n], 0))  # initial (X,Y,Z) nodal coordinates

            # Write element connectivity (3 for triangle, 2 for bar)
            fout.write('\nCELLS %s %s \n' % (nelems, (nelemsMem * 4 + nelemsBar * 3)))

            if nelemsBar != 0:
                for el in range(nelemsBar):
                    fout.write(
                        '2 %s %s\n' % (NBar[el, 1], NBar[el, 2]))  # connectivity for bar
            if nelemsMem != 0:
                for el in range(nelemsMem):
                    fout.write(
                        '3 %s %s %s\n' % (NMem[el, 1], NMem[el, 2], NMem[el, 3]))  # connectivity for triangle

            # Write cell types
            fout.write('\nCELL_TYPES %s \n' % nelems)  # 3 for triangle (3 nodes)
            for n in range(nelemsBar):
                fout.write('3\n')  # 3 = line
            for n in range(nelemsMem):
                fout.write('5\n')  # 5 = triangle

            # Start point data (e.g. displacements)
            fout.write('\nPOINT_DATA %s \n' % nnodes)

            # Write displacements
            writeVectorNode2D(u, nnodes, 'u', fout)

            # Write fixed nodes
            writeVectorNode2D(dofFixedWrite, nnodes, 'fix', fout)

            # Write fixed nodes
            writeVectorNode2D(RF, nnodes, 'reactionF', fout)

            # Start cell data (e.g. strain and stress)
            fout.write('\nCELL_DATA %s \n' % (nelemsBar + nelemsMem))

            # Write strain
            writeVectorElement(Eelastic, 'E', nelemsBar + nelemsMem, fout)

            # Write principal strain
            writeVectorElement(Ep, 'EP', nelemsBar + nelemsMem, fout)

            # Write stress
            writeVectorElement(S, 'S', nelemsBar + nelemsMem, fout)

            # Write principal stress
            writeVectorElement(Sp, 'SP', nelemsBar + nelemsMem, fout)

            # Write Von Mises stress
            writeScalarElement(VMS, "MISES", nelemsBar + nelemsMem, fout)

    print("Finished writing output to {}".format(name))


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int writeVTK3D(object data) except -1:

    cdef:

        str inputFile = data.inputName

        unsigned int nnodes = data.nnodes
        unsigned int nelems = data.nelems
        unsigned int nelemsMem = data.nelemsMem
        unsigned int nelemsBar = data.nelemsBar
        unsigned int ndof = data.ndof
        unsigned int n, el, index

        int time = data.time

        double t =       data.props["t"]
        double areaBar = data.props["areaBar"]
        double rhoMem =  data.props["rhoMem"]
        double rhoBar =  data.props["rhoBar"]
        double EMem =    data.props["EMem"]
        double EBar =    data.props["EBar"]
        double poisson = data.props["nu"]

        int [:] dofFixed = data.dofFixed

        double [:] X0 = data.X0
        double [:] Y0 = data.Y0
        double [:] Z0 = data.Z0
        double [:] u = data.u
        double [:] X = data.X
        double [:] Y = data.Y
        double [:] Z = data.Z
        double [:] J11Vec = data.J11Vec
        double [:] J12Vec = data.J12Vec
        double [:] J22Vec = data.J22Vec
        double [:] thetaVec = data.thetaVec
        double [:] areaVec = data.areaVec
        double [:] L0 = data.LBar
        double [:] dofFixedWrite = np.ones(ndof, dtype=np.double)
        double [:] VMS = np.zeros(nelemsMem + nelemsBar, dtype=np.double)
        double [:] RF = np.zeros(ndof, dtype=np.double)

        double [:, ::1] S = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)
        double [:, ::1] Sp = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)
        double [:, ::1] Ep = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)
        double [:, ::1] Eelastic = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)
        double [:, ::1] Ew = np.zeros((nelemsMem + nelemsBar, 3), dtype=np.double)

        int [:, ::1] NMem = data.Nm
        int [:, ::1] NBar = data.Nb

        unsigned int [:] state = np.zeros(nelemsMem + nelemsBar, dtype=np.uintc)

    for el in dofFixed:
        dofFixedWrite[el] = 0

    # reaction forces
    index = 0
    for el in dofFixed:
        RF[el] = data.RF[index]
        index += 1

    # compute Cauchy stress and strain
    computeStress3D(X, Y, Z, NMem, NBar, nelemsBar, nelemsMem, J11Vec, J12Vec, J22Vec, EMem,
                    poisson, EBar, areaBar, S, Sp, Ep, Eelastic, thetaVec, VMS, areaVec, L0,
                    Ew, state)

    outputDirectory = "output4mem"

    if not os.path.exists(outputDirectory):
           os.makedirs(outputDirectory)
    if time:
        name = "{}/{}_{:0>{}}.vtk".format(outputDirectory, inputFile, time, data.tLength)
    else:
        name = "{}/{}.vtk".format(outputDirectory,inputFile)

    with open(name, 'w') as fout:

            fout.write('# vtk DataFile Version 2.0\nmem4py VTK writer\nASCII\nDATASET '
                       'UNSTRUCTURED_GRID\nPOINTS %s float\n' % nnodes)

            # Write undeformed node coordinates
            for n in range(0, nnodes):
                fout.write('%s %s %s\n' % (X0[n], Y0[n], Z0[n]))  # initial (X,Y,Z) nodal coordinates

            # Write element connectivity (3 for triangle, 2 for bar)
            fout.write('\nCELLS %s %s \n' % (nelems, (nelemsMem * 4 + nelemsBar * 3)))

            if nelemsBar != 0:
                for el in range(nelemsBar):
                    fout.write(
                        '2 %s %s\n' % (NBar[el, 1], NBar[el, 2]))  # connectivity for bar
            if nelemsMem != 0:
                for el in range(nelemsMem):
                    fout.write(
                        '3 %s %s %s\n' % (NMem[el, 1], NMem[el, 2], NMem[el, 3]))  # connectivity for triangle

            # Write cell types
            fout.write('\nCELL_TYPES %s \n' % nelems)  # 3 for triangle (3 nodes)
            for n in range(nelemsBar):
                fout.write('3\n')  # 3 = line
            for n in range(nelemsMem):
                fout.write('5\n')  # 5 = triangle

            # Start point data (e.g. displacements)
            fout.write('\nPOINT_DATA %s \n' % nnodes)

            # Write displacements
            writeVectorNode3D(u, nnodes, 'u', fout)

            # Write fixed nodes
            writeVectorNode3D(dofFixedWrite, nnodes, 'fix', fout)

            # Write fixed nodes
            writeVectorNode3D(RF, nnodes, 'reactionF', fout)

            # Start cell data (e.g. strain and stress)
            fout.write('\nCELL_DATA %s \n' % (nelemsBar + nelemsMem))

            # Write strain
            writeVectorElement(Eelastic, 'E', nelemsBar + nelemsMem, fout)

            # Write principal strain
            writeVectorElement(Ep, 'EP', nelemsBar + nelemsMem, fout)

            # Write stress
            writeVectorElement(S, 'S', nelemsBar + nelemsMem, fout)

            # Write principal stress
            writeVectorElement(Sp, 'SP', nelemsBar + nelemsMem, fout)

            # Write Von Mises stress
            writeScalarElement(VMS, "MISES", nelemsBar + nelemsMem, fout)

    print("Finished writing output to {}".format(name))
