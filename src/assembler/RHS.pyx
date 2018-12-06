import numpy as np
cimport numpy as np
cimport cython

from src.ceygen.math cimport add_vv

cdef extern from "math.h":
    double sqrt(double m)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int RHS3D(double [:] X,
               double [:] Y,
               double [:] Z,
               int [:, ::1] Nm,
               int [:, ::1] Nb,
               double [:] p,
               double [:] RHS,
               double [:] RHS0,
               unsigned int [:] elPressurised,
               double [:] areaVec,
               double [:] L0,
               int gravity,
               unsigned int nelemsMem,
               unsigned int nelemsBar,
               double t,
               double rhoMem,
               double areaBar,
               double rhoBar,
               double [:] g,
               double [:] Sx,
               double [:] Sy,
               double [:] Sz, #) except -1:
               unsigned int [:, ::1] loadedBCNodes,
               unsigned int [:, ::1] loadedBCEdges,
               unsigned int RHS0flag,
               object load,
               double loadStep) except -1:

    """Assemble RHS from all load sources for Dynamic Relaxation solver
    
    assemble RHS0 and RHS. RHS0 is due to constant loads. RHS = RHS0 + RHS(due to follower load)
    
    constant loads:
    
        - fX, fY, fZ              ID [1, 2, 3]
        - edgeX, edgeY, edgeZ     ID [1, 2, 3]
        - gravity [gX, gY, gZ]
        
    follower loads:
    
        - p
        
    """
    cdef:
        unsigned int el, BCNodesLength, nPressurised = len(elPressurised), dof1, dof2
        double fx, fy, fz, crossX, crossY, crossZ, area

        unsigned int [:] allDofBar = np.zeros(6, dtype=np.uintc)
        unsigned int [:] allDofMem = np.zeros(9, dtype=np.uintc)

    # if gravity is on and has not been computed before
    if gravity == 1 and RHS0flag == 0:

        # loop through bar elements
        for el in range(nelemsBar):

            # Find degrees of freedom from current element
            allDofBar[0] = 3 * (Nb[el, 1] + 1) - 3
            allDofBar[1] = 3 * (Nb[el, 1] + 1) - 2
            allDofBar[2] = 3 * (Nb[el, 1] + 1) - 1
            allDofBar[3] = 3 * (Nb[el, 2] + 1) - 3
            allDofBar[4] = 3 * (Nb[el, 2] + 1) - 2
            allDofBar[5] = 3 * (Nb[el, 2] + 1) - 1

            # nodal forces due to gravity (acc * volume * density) for two nodes
            fx = loadStep * g[0] * areaBar * L0[el] * rhoBar / 2.
            fy = loadStep * g[1] * areaBar * L0[el] * rhoBar / 2.
            fz = loadStep * g[2] * areaBar * L0[el] * rhoBar / 2.

            RHS0[allDofBar[0]] += fx
            RHS0[allDofBar[1]] += fy
            RHS0[allDofBar[2]] += fz
            RHS0[allDofBar[3]] += fx
            RHS0[allDofBar[4]] += fy
            RHS0[allDofBar[5]] += fz

        # loop through membrane elements
        for el in range(nelemsMem):

            # Find degrees of freedom from current element
            allDofMem[0] = 3 * (Nm[el, 1] + 1) - 3
            allDofMem[1] = 3 * (Nm[el, 1] + 1) - 2
            allDofMem[2] = 3 * (Nm[el, 1] + 1) - 1
            allDofMem[3] = 3 * (Nm[el, 2] + 1) - 3
            allDofMem[4] = 3 * (Nm[el, 2] + 1) - 2
            allDofMem[5] = 3 * (Nm[el, 2] + 1) - 1
            allDofMem[6] = 3 * (Nm[el, 3] + 1) - 3
            allDofMem[7] = 3 * (Nm[el, 3] + 1) - 2
            allDofMem[8] = 3 * (Nm[el, 3] + 1) - 1

            # Test for comparison
            fx = loadStep * g[0] * areaVec[el] * t * rhoMem / 3
            fy = loadStep * g[1] * areaVec[el] * t * rhoMem / 3
            fz = loadStep * g[2] * areaVec[el] * t * rhoMem / 3

            # Directly insert forces into RHS vector (slow method?)
            RHS0[allDofMem[0]] += fx
            RHS0[allDofMem[1]] += fy
            RHS0[allDofMem[2]] += fz
            RHS0[allDofMem[3]] += fx
            RHS0[allDofMem[4]] += fy
            RHS0[allDofMem[5]] += fz
            RHS0[allDofMem[6]] += fx
            RHS0[allDofMem[7]] += fy
            RHS0[allDofMem[8]] += fz

    # Compute nodal load contributions (fX, fY, fZ)
    if RHS0flag == 0 and loadedBCNodes[0, 0] != 0:

        BCNodesLength = np.size(loadedBCNodes, 0)

        # loop through nodal loads
        for el in range(BCNodesLength):

            if loadedBCNodes[el, 0] == 1:
                dof1 = 3 * (loadedBCNodes[el, 1] + 1) - 3
                RHS0[dof1] += loadStep * load["fX"]

            elif loadedBCNodes[el, 0] == 2:
                dof1 = 3 * (loadedBCNodes[el, 1] + 1) - 2
                RHS0[dof1] += loadStep * load["fY"]

            elif loadedBCNodes[el, 0] == 3:
                dof1 = 3 * (loadedBCNodes[el, 1] + 1) - 1
                RHS0[dof1] += loadStep * load["fZ"]

    # Compute edge load contributions (edgeX, edgeY, edgeZ, edgeNormal)
    if loadedBCEdges[0, 0] != 0:

        BCNodesLength = np.size(loadedBCEdges, 0)

        # loop through nodal loads
        for el in range(BCNodesLength):

            l = sqrt((X[loadedBCEdges[el, 2]] - X[loadedBCEdges[el, 1]]) ** 2 +
                     (Y[loadedBCEdges[el, 2]] - Y[loadedBCEdges[el, 1]]) ** 2 +
                     (Z[loadedBCEdges[el, 2]] - Z[loadedBCEdges[el, 1]]) ** 2)

            if loadedBCEdges[el, 0] == 1: # edgeX

                dof1 = 3 * (loadedBCEdges[el, 1] + 1) - 3
                dof2 = 3 * (loadedBCEdges[el, 2] + 1) - 3

                RHS0[dof1] += l * t * loadStep * load["edgeX"] / 2
                RHS0[dof2] += l * t * loadStep * load["edgeX"] / 2

            elif loadedBCEdges[el, 0] == 2: # edgeY

                dof1 = 3 * (loadedBCEdges[el, 1] + 1) - 2
                dof2 = 3 * (loadedBCEdges[el, 2] + 1) - 2

                RHS0[dof1] += l * t * loadStep * load["edgeY"] / 2
                RHS0[dof2] += l * t * loadStep * load["edgeY"] / 2

            elif loadedBCEdges[el, 0] == 3: # edgeZ

                dof1 = 3 * (loadedBCEdges[el, 1] + 1) - 1
                dof2 = 3 * (loadedBCEdges[el, 2] + 1) - 1

                RHS0[dof1] += l * t * loadStep * load["edgeZ"] / 2
                RHS0[dof2] += l * t * loadStep * load["edgeZ"] / 2

    # Initialise RHS
    RHS[...] = 0

    # Loop through all pressurised elements
    for el in range(nPressurised):

        # Find degrees of freedom from current element
        allDofMem[0] = 3 * (Nm[elPressurised[el], 1] + 1) - 3
        allDofMem[1] = 3 * (Nm[elPressurised[el], 1] + 1) - 2
        allDofMem[2] = 3 * (Nm[elPressurised[el], 1] + 1) - 1
        allDofMem[3] = 3 * (Nm[elPressurised[el], 2] + 1) - 3
        allDofMem[4] = 3 * (Nm[elPressurised[el], 2] + 1) - 2
        allDofMem[5] = 3 * (Nm[elPressurised[el], 2] + 1) - 1
        allDofMem[6] = 3 * (Nm[elPressurised[el], 3] + 1) - 3
        allDofMem[7] = 3 * (Nm[elPressurised[el], 3] + 1) - 2
        allDofMem[8] = 3 * (Nm[elPressurised[el], 3] + 1) - 1

        # cross product components of current CST element
        crossX = (Y[Nm[elPressurised[el], 2]] - Y[Nm[elPressurised[el], 1]]) * \
                 (Z[Nm[elPressurised[el], 3]] - Z[Nm[elPressurised[el], 1]]) - \
                 (Y[Nm[elPressurised[el], 3]] - Y[Nm[elPressurised[el], 1]]) * \
                 (Z[Nm[elPressurised[el], 2]] - Z[Nm[elPressurised[el], 1]])

        crossY = (X[Nm[elPressurised[el], 3]] - X[Nm[elPressurised[el], 1]]) * \
                 (Z[Nm[elPressurised[el], 2]] - Z[Nm[elPressurised[el], 1]]) - \
                 (X[Nm[elPressurised[el], 2]] - X[Nm[elPressurised[el], 1]]) * \
                 (Z[Nm[elPressurised[el], 3]] - Z[Nm[elPressurised[el], 1]])

        crossZ = (X[Nm[elPressurised[el], 2]] - X[Nm[elPressurised[el], 1]]) * \
                 (Y[Nm[elPressurised[el], 3]] - Y[Nm[elPressurised[el], 1]]) - \
                 (X[Nm[elPressurised[el], 3]] - X[Nm[elPressurised[el], 1]]) * \
                 (Y[Nm[elPressurised[el], 2]] - Y[Nm[elPressurised[el], 1]])

        # current area
        area = sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ)

        # force components (pressure = p/6 * cross, stressX = Sx * A / 3)
        fx = loadStep * (crossX * p[el] + 2 * Sx[el] * area) / 6
        fy = loadStep * (crossY * p[el] + 2 * Sy[el] * area) / 6
        fz = loadStep * (crossZ * p[el] + 2 * Sz[el] * area) / 6

        # Directly insert forces into RHS vector
        RHS[allDofMem[0]] += fx
        RHS[allDofMem[1]] += fy
        RHS[allDofMem[2]] += fz
        RHS[allDofMem[3]] += fx
        RHS[allDofMem[4]] += fy
        RHS[allDofMem[5]] += fz
        RHS[allDofMem[6]] += fx
        RHS[allDofMem[7]] += fy
        RHS[allDofMem[8]] += fz

    # if gravity is on and was computed, add it to pressure contribution
    if RHS0flag == 1:
        add_vv(RHS, RHS0, RHS)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int RHS2D(double [:] X,
               double [:] Y,
               int [:, ::1] Nm,
               int [:, ::1] Nb,
               double [:] RHS,
               double [:] areaVec,
               double [:] L0,
               int gravity,
               unsigned int nelemsMem,
               unsigned int nelemsBar,
               double t,
               double rhoMem,
               double areaBar,
               double rhoBar,
               double [:] g,
               unsigned int [:, ::1] loadedBCNodes,
               unsigned int [:, ::1] loadedBCEdges,
               object load,
               double loadStep) except -1:

    """Assemble RHS from all load sources for Dynamic Relaxation solver
    
    assemble RHS0 and RHS. RHS0 is due to constant loads. RHS = RHS0 + RHS(due to follower load)
    
    constant loads:
    
        - fX, fY, fZ              ID [1, 2, 3]
        - edgeX, edgeY, edgeZ     ID [1, 2, 3]
        - gravity [gX, gY, gZ]
        
    follower loads:
    
        - p
        
    """
    cdef:
        unsigned int el, BCNodesLength, dof1, dof2
        double fx, fy

        unsigned int [:] allDofBar = np.zeros(4, dtype=np.uintc)
        unsigned int [:] allDofMem = np.zeros(6, dtype=np.uintc)

    # if gravity is on and has not been computed before
    if gravity == 1:

        # loop through bar elements
        for el in range(nelemsBar):

            # Find degrees of freedom from current element
            allDofBar[0] = 2 * (Nb[el, 1] + 1) - 2
            allDofBar[1] = 2 * (Nb[el, 1] + 1) - 1
            allDofBar[2] = 2 * (Nb[el, 2] + 1) - 2
            allDofBar[3] = 2 * (Nb[el, 2] + 1) - 1

            # nodal forces due to gravity (acc * volume * density) for two nodes
            fx = loadStep * g[0] * areaBar * L0[el] * rhoBar / 2.
            fy = loadStep * g[1] * areaBar * L0[el] * rhoBar / 2.

            RHS[allDofBar[0]] += fx
            RHS[allDofBar[1]] += fy
            RHS[allDofBar[2]] += fx
            RHS[allDofBar[3]] += fy

        # loop through membrane elements
        for el in range(nelemsMem):

            # Find degrees of freedom from current element
            allDofMem[0] = 2 * (Nm[el, 1] + 1) - 2
            allDofMem[1] = 2 * (Nm[el, 1] + 1) - 1
            allDofMem[2] = 2 * (Nm[el, 2] + 1) - 2
            allDofMem[3] = 2 * (Nm[el, 2] + 1) - 1
            allDofMem[4] = 2 * (Nm[el, 3] + 1) - 2
            allDofMem[5] = 2 * (Nm[el, 3] + 1) - 1

            # Test for comparison
            fx = loadStep * g[0] * areaVec[el] * t * rhoMem / 3
            fy = loadStep * g[1] * areaVec[el] * t * rhoMem / 3

            # Directly insert forces into RHS vector
            RHS[allDofMem[0]] += fx
            RHS[allDofMem[1]] += fy
            RHS[allDofMem[2]] += fx
            RHS[allDofMem[3]] += fy
            RHS[allDofMem[4]] += fx
            RHS[allDofMem[5]] += fy

    # Compute nodal load contributions (fX, fY)
    if loadedBCNodes[0, 0] != 0:

        BCNodesLength = np.size(loadedBCNodes, 0)

        # loop through nodal loads
        for el in range(BCNodesLength):

            if loadedBCNodes[el, 0] == 1:
                dof1 = 2 * (loadedBCNodes[el, 1] + 1) - 2
                RHS[dof1] += loadStep * load["fX"]

            elif loadedBCNodes[el, 0] == 2:
                dof1 = 2 * (loadedBCNodes[el, 1] + 1) - 1
                RHS[dof1] += loadStep * load["fY"]

    # Compute edge load contributions (edgeX, edgeY)
    if loadedBCEdges[0, 0] != 0:

        BCNodesLength = np.size(loadedBCEdges, 0)

        # loop through nodal loads
        for el in range(BCNodesLength):

            l = sqrt((X[loadedBCEdges[el, 2]] - X[loadedBCEdges[el, 1]]) ** 2 +
                     (Y[loadedBCEdges[el, 2]] - Y[loadedBCEdges[el, 1]]) ** 2)

            if loadedBCEdges[el, 0] == 1: # edgeX

                dof1 = 2 * (loadedBCEdges[el, 1] + 1) - 2
                dof2 = 2 * (loadedBCEdges[el, 2] + 1) - 2

                RHS[dof1] += loadStep * l * t * load["edgeX"] / 2
                RHS[dof2] += loadStep * l * t * load["edgeX"] / 2

            elif loadedBCEdges[el, 0] == 2: # edgeY

                dof1 = 2 * (loadedBCEdges[el, 1] + 1) - 1
                dof2 = 2 * (loadedBCEdges[el, 2] + 1) - 1

                RHS[dof1] += loadStep * l * t * load["edgeY"] / 2
                RHS[dof2] += loadStep * l * t * load["edgeY"] / 2