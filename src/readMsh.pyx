import numpy as np
cimport numpy as np
cimport cython

from src.helper.dirichletHandler cimport initialiseDirichletBC
from src.helper.area cimport area

cdef extern from "math.h":
    double sqrt(double m)
    double atan2(double m, double n)
    double cos(double m)
    double sin(double m)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(True)  # turn off negative index wrapping for entire function
cdef int readMsh(object data) except -1:
    """
    readMsh function reads .msh file and prepares nodal and element information
    :param data: mem4py main object with all necessary info
    :return: errorCode
    
    How the load definition work:
    
    constant load:
    
        - fX, fY, fZ
        - edgeX, edgeY, edgeZ
        - gravity gX, gY, gZ
        
    follower load:
    
        - p
        - edgeNormal
        
    node ID:
    fX = 1
    fY = 2
    fZ = 3
    
    line ID:
    "edgeX" = 1
    "edgeY" = 2
    "edgeZ" = 3
    "edgeNormal" = 4
    
    """
    # Initialise variables
    state = ""
    readArgs = True
    fixedBCNodesRead = []
    loadedBCNodesRead = []
    loadedBCEdgesRead = []
    prescribedBCNodesRead = []
    bnames = dict()
    Xread = []
    Yread = []
    Zread = []
    NBarRead = []
    NMemRead = []

    cdef int nnodes = 0

    with open("msh/{}.msh".format(data.inputName)) as fin:
        for line in fin:
            if 'MeshFormat' in line:
                state = 'MeshFormat'
                readArgs = True
                continue
            elif 'PhysicalNames' in line:
                state = 'PhysicalNames'
                readArgs = True
                continue
            elif 'Nodes' in line:
                state = 'Nodes'
                readArgs = True
                continue
            elif 'Elements' in line:
                state = 'Elements'
                readArgs = True
                continue
            if readArgs:
                if state == 'MeshFormat':
                    args = line.split()
                    if float(args[0]) >= 4:
                        raise Exception("Cannot read msh input file version 4.0. Use -format msh2 when creating msh file")
                    readArgs = False
                elif state == 'PhysicalNames':
                    nboundaries = int(line)
                    readArgs = False
                    continue
                elif state == 'Nodes':
                    nnodes = int(line)
                    readArgs = False
                    continue
                elif state == 'Elements':
                    nconditions = int(line)
                    readArgs = False
                    continue
            else:
                if state == 'PhysicalNames':
                    # dim, id, name
                    args = line.split()
                    name = args[2].replace('"', '')
                    bnames[int(args[1])] = name
                elif state == 'Nodes':
                    # Nodal coordinates in vectors X, Y, Z
                    args = line.split()
                    Xread.append(float(args[1]) * data.scale)
                    Yread.append(float(args[2]) * data.scale)
                    Zread.append(float(args[3]) * data.scale)
                elif state == 'Elements':
                    # elementNumber, elementType, numberOfTags, physicalID, elementaryID, nodeCon
                    #
                    # elementType:
                    # 1 = 2-node line
                    # 2 = 3-node triangle
                    # 8 = 3-node second order line
                    # 9 = 6-node second order triangle
                    # 15 = 1-node point
                    #
                    # numberOfTags:
                    # gives the number of integer tags that follow for current element.
                    # - First tag is always the physical entity (in this case the real element description as
                    # defined in bnames
                    # - Second tag is the elementary tag (currently not used)
                    #
                    # Last entries give node connectivity of element

                    args = line.split()

                    if int(args[1]) == 15:    # 1-node point
                        BCcase = bnames[int(args[3])]
                        if BCcase == "fixAll":
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixX":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixY":
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixZ":
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixXY":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixXZ":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixYZ":
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fX":
                            loadedBCNodesRead.append([1, int(args[-1]) - 1])
                        elif BCcase == "fY":
                            loadedBCNodesRead.append([2, int(args[-1]) - 1])
                        elif BCcase == "fZ":
                            loadedBCNodesRead.append([3, int(args[-1]) - 1])
                        elif BCcase == "u0":
                            prescribedBCNodesRead.append([1, int(args[-1]) - 1])
                        elif BCcase == "v0":
                            prescribedBCNodesRead.append([2, int(args[-1]) - 1])
                        elif BCcase == "w0":
                            prescribedBCNodesRead.append([3, int(args[-1]) - 1])
                        else:
                            raise Exception("Unspecified BC definition found in 1-node point", int(args[0]))
                    elif int(args[1]) == 1:    # 2-nodes line
                        BCcase = bnames[int(args[3])]
                        if BCcase == "fixAll":
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixX":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixY":
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixZ":
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixXY":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixXZ":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixYZ":
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "edgeX":
                            loadedBCEdgesRead.append([1, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "edgeY":
                            loadedBCEdgesRead.append([2, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "edgeZ":
                            loadedBCEdgesRead.append([3, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "edgeNormal":
                            loadedBCEdgesRead.append([4, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "FSI":
                            loadedBCEdgesRead.append([5, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "CpTop":
                            loadedBCEdgesRead.append([6, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "CpBot":
                            loadedBCEdgesRead.append([7, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "u0":
                            prescribedBCNodesRead.append([1, int(args[-2]) - 1])
                            prescribedBCNodesRead.append([1, int(args[-1]) - 1])
                        elif BCcase == "v0":
                            prescribedBCNodesRead.append([2, int(args[-2]) - 1])
                            prescribedBCNodesRead.append([2, int(args[-1]) - 1])
                        elif BCcase == "w0":
                            prescribedBCNodesRead.append([3, int(args[-2]) - 1])
                            prescribedBCNodesRead.append([3, int(args[-1]) - 1])
                        elif BCcase == "bar":    # N = [ID, N1, N2, -1]
                            NBarRead.append([0, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "beam":    # N = [ID, N1, N2, -1]
                            NBarRead.append([1, int(args[-2]) - 1, int(args[-1]) - 1])
                        else:
                            raise Exception("Unspecified BC definition found in 2-node line element", int(args[0]))
                    elif int(args[1]) == 2:    # 3-nodes triangle
                        BCcase = bnames[int(args[3])]
                        if BCcase == "membrane":
                            NMemRead.append([0, int(args[-3]) - 1, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "pMembrane":
                            NMemRead.append([1, int(args[-3]) - 1, int(args[-2]) - 1, int(args[-1]) - 1])
                        elif BCcase == "fixAll":
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(0)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixX":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixY":
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixZ":
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixXY":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixYZ":
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(2)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        elif BCcase == "fixXZ":
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(1)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-3]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-2]) - 1)
                            fixedBCNodesRead.append(3)
                            fixedBCNodesRead.append(int(args[-1]) - 1)
                        else:
                            raise Exception("Unspecified physical surface identifier found in 3-node triangle",
                                  int(args[0]))
                    else:
                        raise Exception("Unspecified element identifier found in", int(args[0]))

    # Define nodal coordinates and connectivity as C arrays
    cdef double [:] X0 = np.asarray(Xread)
    cdef double [:] Y0 = np.asarray(Yread)
    cdef double [:] Z0 = np.asarray(Zread)
    NBar = np.asarray(NBarRead, dtype=np.intc)
    NMem = np.asarray(NMemRead, dtype=np.intc)
    if NBar.size == 0:
        NBar = np.array([[-1, 0, 0]], dtype=np.intc)
    if NMem.size == 0:
        NMem = np.array([[-1, 0, 0, 0]], dtype=np.intc)
    cdef int [:, ::1] Nb = NBar
    cdef int [:, ::1] Nm = NMem

    cdef unsigned int nelemsMem = len(np.where(NMem[:, 0] == 0)[0]) + len(np.where(NMem[:, 0] == 1)[0])
    cdef unsigned int nPressurised = len(np.where(NMem[:, 0] == 1)[0])
    cdef unsigned int nelemsBar = len(np.where(NBar[:, 0] == 0)[0])
    cdef unsigned int nelemsBeam = len(np.where(NBar[:, 0] == 1)[0])
    cdef unsigned int nelems = np.sum((nelemsBar, nelemsBeam, nelemsMem))

    # Fixed nodes as C array
    cdef unsigned int [:] fixedBCNodes = np.asarray(fixedBCNodesRead, dtype=np.uintc)
    cdef unsigned int el

    # Call dirichletHandler for dofFree
    dofFixedP = initialiseDirichletBC(fixedBCNodes, data.dim)

    if not loadedBCNodesRead:
        loadedBCNodesRead = np.zeros((1, 2), dtype=np.uintc)
    else:
        loadedBCNodesRead = np.asarray(loadedBCNodesRead)
        loadedBCNodesRead = loadedBCNodesRead[loadedBCNodesRead[:,0].argsort()]
    cdef unsigned int [:, ::1] loadedBCNodes = np.asarray(loadedBCNodesRead, dtype=np.uintc)

    if not loadedBCEdgesRead:
        loadedBCEdgesRead = np.zeros((1, 3), dtype=np.uintc)
    else:
        loadedBCEdgesRead = np.asarray(loadedBCEdgesRead)
        loadedBCEdgesRead = loadedBCEdgesRead[loadedBCEdgesRead[:,0].argsort()]
    cdef unsigned int [:, ::1] loadedBCEdges = np.asarray(loadedBCEdgesRead, dtype=np.uintc)

    if prescribedBCNodesRead:
        prescribedBCNodesRead = np.unique(prescribedBCNodesRead, axis=0)
    else:
        prescribedBCNodesRead = - np.ones((1, 1), dtype=np.intc)

    cdef int [:] prescribedDof = np.empty(prescribedBCNodesRead.shape[0], dtype=np.intc)
    cdef double [:] prescribedDisplacement = np.zeros(prescribedBCNodesRead.shape[0], dtype=np.double)

    if prescribedBCNodesRead[0, 0] == -1:
        prescribedDof[0] = -1
    else:
        for el in range(prescribedBCNodesRead.shape[0]):
            if prescribedBCNodesRead[el, 0] == 1:
                prescribedDof[el] = data.dim * (prescribedBCNodesRead[el, 1] + 1) - data.dim
            elif prescribedBCNodesRead[el, 0] == 2:
                prescribedDof[el] = data.dim * (prescribedBCNodesRead[el, 1] + 1) - data.dim + 1
            else:
                prescribedDof[el] = data.dim * (prescribedBCNodesRead[el, 1] + 1) - data.dim + 2
        ind = np.argsort(prescribedDof)

        for el in range(len(ind)):
            prescribedDisplacement[el] = prescribedBCNodesRead[ind[el], 0]
        prescribedDof = np.sort(prescribedDof)

    cdef int [:] dofFixed = dofFixedP.astype(np.intc)
    cdef unsigned int [:] elPressurised = np.where(NMem[:, 0] == 1)[0].astype(np.uintc)
    cdef double [:] p = np.ones(len(elPressurised), dtype=np.double) * data.load["p"]

    # distributed load from OpenFOAM
    cdef double [:] Sx = np.zeros(len(elPressurised), dtype=np.double)
    cdef double [:] Sy = np.zeros(len(elPressurised), dtype=np.double)
    cdef double [:] Sz = np.zeros(len(elPressurised), dtype=np.double)

    # cell centres
    cdef double [:, ::1] cc = np.zeros((len(elPressurised), 3), dtype=np.double)

    # area
    cdef double [:] areaVec = np.empty(nelemsMem, dtype=np.double)
    area(X0, Y0, Z0, Nm, nelemsMem, areaVec)
    data.areaVec = areaVec

    # determine parts of the deformation gradient
    cdef double [:] E3 = np.zeros(3, dtype=np.double)
    cdef double [:] E2 = np.zeros(3, dtype=np.double)
    cdef double [:] E1 = np.zeros(3, dtype=np.double)
    cdef double [:] normalVecX = np.zeros(nelemsMem, dtype=np.double)
    cdef double [:] normalVecY = np.zeros(nelemsMem, dtype=np.double)
    cdef double [:] normalVecZ = np.zeros(nelemsMem, dtype=np.double)
    cdef double [:] J11Vec = np.zeros(nelemsMem, dtype=np.double)
    cdef double [:] J12Vec = np.zeros(nelemsMem, dtype=np.double)
    cdef double [:] J22Vec = np.zeros(nelemsMem, dtype=np.double)
    cdef double [:] thetaVec = np.zeros(nelemsMem, dtype=np.double)
    cdef double [:] thetaCylinder = np.zeros(nelemsMem, dtype=np.double)
    cdef double Nnorm

    cdef double [:] LBar = np.empty(nelemsBar, dtype=np.double)
    cdef double [:] ed = np.empty(3, dtype=np.double)
    cdef double [:, ::1] R = np.empty((3, 3), dtype=np.double)

    # Material orientation
    matDir = data.matDir
    if data.matDirAngle:
        matDirAngle = data.matDirAngle
    else:
        matDirAngle = 0

    # bars
    for el in range(nelemsBar):

        LBar[el] = sqrt((X0[Nb[el, 2]] - X0[Nb[el, 1]]) * (X0[Nb[el, 2]] - X0[Nb[el, 1]]) +
                        (Y0[Nb[el, 2]] - Y0[Nb[el, 1]]) * (Y0[Nb[el, 2]] - Y0[Nb[el, 1]]) +
                        (Z0[Nb[el, 2]] - Z0[Nb[el, 1]]) * (Z0[Nb[el, 2]] - Z0[Nb[el, 1]]))

    # membranes
    for el in range(nelemsMem):

        J11Vec[el] = sqrt((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (X0[Nm[el, 2]] - X0[Nm[el, 1]]) +
                          (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) +
                          (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]))
        J12Vec[el] = ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) +
                      (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) +
                      (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) / J11Vec[el]

        Nmnorm = sqrt(((Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) -
                      (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]])) *
                     ((Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) -
                      (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]])) +

                     ((Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) -
                      (X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) *
                     ((Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) -
                      (X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) +

                     ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) -
                      (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]])) *
                     ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) -
                      (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]])))

        normalVecX[el] = ((Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) -
                      (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]])) / Nmnorm
        normalVecY[el] = ((Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) -
                      (X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) / Nmnorm
        normalVecZ[el] = ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) -
                      (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]])) / Nmnorm

        # unit vector of X12
        E1[0] = (X0[Nm[el, 2]] - X0[Nm[el, 1]]) / J11Vec[el]
        E1[1] = (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) / J11Vec[el]
        E1[2] = (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) / J11Vec[el]

        # normal (unit) vector to triangle surface
        E3[0] = ((Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) -
                (Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]])) / (2 * areaVec[el])
        E3[1] = ((Z0[Nm[el, 2]] - Z0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]]) -
                (X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Z0[Nm[el, 3]] - Z0[Nm[el, 1]])) / (2 * areaVec[el])
        E3[2] = ((X0[Nm[el, 2]] - X0[Nm[el, 1]]) * (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) -
                (Y0[Nm[el, 2]] - Y0[Nm[el, 1]]) * (X0[Nm[el, 3]] - X0[Nm[el, 1]])) / (2 * areaVec[el])

        # unit vector (N x E1)
        E2[0] = E3[1] * E1[2] - E3[2] * E1[1]
        E2[1] = E3[2] * E1[0] - E3[0] * E1[2]
        E2[2] = E3[0] * E1[1] - E3[1] * E1[0]

        J22Vec[el] = (X0[Nm[el, 3]] - X0[Nm[el, 1]]) * E2[0] + \
                     (Y0[Nm[el, 3]] - Y0[Nm[el, 1]]) * E2[1] + \
                     (Z0[Nm[el, 3]] - Z0[Nm[el, 1]]) * E2[2]

        # print("J22Vec[el] = {}".format(J22Vec[el]))
        # test = sqrt(((Y0[N[el, 2]] - Y0[N[el, 1]]) * (Z0[N[el, 3]] - Z0[N[el, 1]]) -
        #              (Z0[N[el, 2]] - Z0[N[el, 1]]) * (Y0[N[el, 3]] - Y0[N[el, 1]])) ** 2 +
        # ((Z0[N[el, 2]] - Z0[N[el, 1]]) * (X0[N[el, 3]] - X0[N[el, 1]]) -
        #              (X0[N[el, 2]] - X0[N[el, 1]]) * (Z0[N[el, 3]] - Z0[N[el, 1]])) ** 2 +
        # ((X0[N[el, 2]] - X0[N[el, 1]]) * (Y0[N[el, 3]] - Y0[N[el, 1]]) -
        #              (Y0[N[el, 2]] - Y0[N[el, 1]]) * (X0[N[el, 3]] - X0[N[el, 1]])) ** 2) / J11Vec[el]
        #
        # print("J22VecCopy = {}".format(test))

        # theta
        thetaVec[el] = atan2(np.dot(E2, matDir), np.dot(E1, matDir))

        # Rodrigues rotation matrix
        R[0, 0] = cos(matDirAngle) + normalVecX[el] * normalVecX[el] * (1 - cos(matDirAngle))
        R[0, 1] = normalVecX[el] * normalVecY[el] * (1 - cos(matDirAngle)) - \
                  normalVecZ[el] * sin(matDirAngle)
        R[0, 2] = normalVecY[el] * sin(matDirAngle) + \
                  normalVecX[el] * normalVecZ[el] * (1 - cos(matDirAngle))

        R[1, 0] = normalVecZ[el] * sin(matDirAngle) + \
                  normalVecX[el] * normalVecY[el] * (1 - cos(matDirAngle))
        R[1, 1] = cos(matDirAngle) + normalVecY[el] * normalVecY[el] * (1 - cos(matDirAngle))
        R[1, 2] = normalVecY[el] * normalVecZ[el] * (1 - cos(matDirAngle)) - \
                  normalVecX[el] * sin(matDirAngle)

        R[2, 0] = normalVecX[el] * normalVecZ[el] * (1 - cos(matDirAngle)) - \
                  normalVecY[el] * sin(matDirAngle)
        R[2, 1] = normalVecX[el] * sin(matDirAngle) + \
                  normalVecY[el] * normalVecZ[el] * (1 - cos(matDirAngle))
        R[2, 2] = cos(matDirAngle) + normalVecZ[el] * normalVecZ[el] * (1 - cos(matDirAngle))

        # find ed material unit vector on cylinder
        ed = np.matmul(R, matDir)

        # local material angle
        thetaCylinder[el] = atan2(np.dot(E2, ed), np.dot(E1, ed))

    data.LBar = LBar

    if data.matDirType == "global":
        data.thetaVec = thetaVec
    elif data.matDirType == "specific":
        print("nothing there yet")
        quit(1)
    elif data.matDirType == "cylinder":
        # cylinder axis
        data.thetaVec = thetaCylinder
    else:
        raise Exception("Specify matDirType to either global or specific")

    data.J11Vec = J11Vec
    data.J12Vec = J12Vec
    data.J22Vec = J22Vec

    data.normalVecX = normalVecX
    data.normalVecY = normalVecY
    data.normalVecZ = normalVecZ

    data.nnodes = nnodes
    data.nelems = nelems
    data.ndof = data.dim * nnodes
    data.nelemsMem = nelemsMem
    data.nelemsBar = nelemsBar

    data.X0 = X0
    data.Y0 = Y0
    data.Z0 = Z0
    cdef double [:] X = np.copy(X0)
    cdef double [:] Y = np.copy(Y0)
    cdef double [:] Z = np.copy(Z0)
    data.X = X
    data.Y = Y
    data.Z = Z

    cdef double [:] u = np.zeros(data.ndof, dtype=np.double)
    cdef double [:] V = np.zeros(data.ndof, dtype=np.double)
    data.u = u
    data.V = V

    data.Nb = Nb
    data.Nm = Nm

    data.dofFixed = dofFixed
    data.prescribedDof = prescribedDof
    data.prescribedDisplacement = prescribedDisplacement
    data.loadedBCNodes = loadedBCNodes
    data.loadedBCEdges = loadedBCEdges
    data.elPressurised = elPressurised
    data.nPressurised = nPressurised
    data.p = p

    data.Sx = Sx
    data.Sy = Sy
    data.Sz = Sz

    data.cc = cc