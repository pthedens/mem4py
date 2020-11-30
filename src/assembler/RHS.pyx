# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
cimport cython

from src.ceygen.ceygenMath cimport add_vv
from src.ceygen.ceygenMath cimport multiply_vs

cdef extern from "math.h":
    double sqrt(double m)
    double fabs(double m)
    double sin(double m)
    double cos(double m)


cdef int assembleRHS(double [:] X,
                     double [:] Y,
                     double [:] Z,
                     double [:] pre_u,
                     int [:, ::1] NMem,
                     int [:, ::1] NCable,
                     double [:] p,
                     double [:] RHS,
                     double [:] RHS0,
                     unsigned int [:] elPressurised,
                     unsigned int [:] elFSI,
                     double [:] area3,
                     double [:] L0,
                     int gravity,
                     unsigned int nelemsMem,
                     unsigned int nelemsCable,
                     unsigned int nPressurised,
                     unsigned int nFSI,
                     double [:] t,
                     double [:] rho3,
                     double [:] area2,
                     double [:] rho2,
                     double [:] g,
                     double [:] Sx,
                     double [:] Sy,
                     double [:] Sz,
                     double [:] pFSI,
                     double [:, ::1] loadedBCNodes,
                     double [:, ::1] loadedBCEdges,
                     unsigned int RHS0flag,
                     double loadStep,
                     unsigned int dim,
                     double [:] force_vector,
                     double [:] E2,
                     double [:] pre_stress_cable,
                     double [:] pre_strain_cable,
                     double [:, ::1] pre_stress_membrane,
                     double [:, ::1] pre_strain_membrane,
                     unsigned int [:] pre_active,
                     long double [:] J11Vec,
                     long double [:] J22Vec,
                     long double [:] J12Vec,
                     double [:] thetaVec,
                     double [:] E3,
                     double [:] nu) except -1:
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
        Py_ssize_t el, dof1, dof2
        double fx, fy, fz, crossX, crossY, crossZ, area, l, dx, dy

        unsigned int [:] allDofCable = np.zeros(2 * dim, dtype=np.uintc)
        unsigned int [:] allDofMem = np.zeros(3 * dim, dtype=np.uintc)

    RHS[...] = 0

    # if gravity is on and has not been computed before
    if gravity == 1 and RHS0flag == 0:

        if dim == 2:

            # loop through cable elements
            for el in range(nelemsCable):

                # Find degrees of freedom from current element
                allDofCable[0] = 2 * (NCable[el, 1] + 1) - 2
                allDofCable[1] = 2 * (NCable[el, 1] + 1) - 1
                allDofCable[2] = 2 * (NCable[el, 2] + 1) - 2
                allDofCable[3] = 2 * (NCable[el, 2] + 1) - 1

                # nodal forces due to gravity (acc * volume * density) for two nodes
                fx = loadStep * g[0] * area2[el] * L0[el] * rho2[el] / 2.
                fy = loadStep * g[1] * area2[el] * L0[el] * rho2[el] / 2.

                RHS0[allDofCable[0]] += fx
                RHS0[allDofCable[1]] += fy
                RHS0[allDofCable[2]] += fx
                RHS0[allDofCable[3]] += fy

            # loop through membrane elements
            for el in range(nelemsMem):

                # Find degrees of freedom from current element
                allDofMem[0] = 2 * (NMem[el, 1] + 1) - 2
                allDofMem[1] = 2 * (NMem[el, 1] + 1) - 1
                allDofMem[2] = 2 * (NMem[el, 2] + 1) - 2
                allDofMem[3] = 2 * (NMem[el, 2] + 1) - 1
                allDofMem[4] = 2 * (NMem[el, 3] + 1) - 2
                allDofMem[5] = 2 * (NMem[el, 3] + 1) - 1

                # Test for comparison
                fx = loadStep * g[0] * area3[el] * t[el] * rho3[el] / 3.
                fy = loadStep * g[1] * area3[el] * t[el] * rho3[el] / 3.

                # Directly insert forces into RHS vector
                RHS0[allDofMem[0]] += fx
                RHS0[allDofMem[1]] += fy
                RHS0[allDofMem[2]] += fx
                RHS0[allDofMem[3]] += fy
                RHS0[allDofMem[4]] += fx
                RHS0[allDofMem[5]] += fy

        elif dim == 3:

            # loop through cable elements
            for el in range(nelemsCable):

                # Find degrees of freedom from current element
                allDofCable[0] = 3 * (NCable[el, 1] + 1) - 3
                allDofCable[1] = 3 * (NCable[el, 1] + 1) - 2
                allDofCable[2] = 3 * (NCable[el, 1] + 1) - 1
                allDofCable[3] = 3 * (NCable[el, 2] + 1) - 3
                allDofCable[4] = 3 * (NCable[el, 2] + 1) - 2
                allDofCable[5] = 3 * (NCable[el, 2] + 1) - 1

                # nodal forces due to gravity (acc * volume * density) for two nodes
                fx = loadStep * g[0] * area2[el] * L0[el] * rho2[el] / 2.
                fy = loadStep * g[1] * area2[el] * L0[el] * rho2[el] / 2.
                fz = loadStep * g[2] * area2[el] * L0[el] * rho2[el] / 2.

                RHS0[allDofCable[0]] += fx
                RHS0[allDofCable[1]] += fy
                RHS0[allDofCable[2]] += fz
                RHS0[allDofCable[3]] += fx
                RHS0[allDofCable[4]] += fy
                RHS0[allDofCable[5]] += fz

            # loop through membrane elements
            for el in range(nelemsMem):

                # Find degrees of freedom from current element
                allDofMem[0] = 3 * (NMem[el, 1] + 1) - 3
                allDofMem[1] = 3 * (NMem[el, 1] + 1) - 2
                allDofMem[2] = 3 * (NMem[el, 1] + 1) - 1
                allDofMem[3] = 3 * (NMem[el, 2] + 1) - 3
                allDofMem[4] = 3 * (NMem[el, 2] + 1) - 2
                allDofMem[5] = 3 * (NMem[el, 2] + 1) - 1
                allDofMem[6] = 3 * (NMem[el, 3] + 1) - 3
                allDofMem[7] = 3 * (NMem[el, 3] + 1) - 2
                allDofMem[8] = 3 * (NMem[el, 3] + 1) - 1

                # Test for comparison
                fx = loadStep * g[0] * area3[el] * t[el] * rho3[el] / 3.
                fy = loadStep * g[1] * area3[el] * t[el] * rho3[el] / 3.
                fz = loadStep * g[2] * area3[el] * t[el] * rho3[el] / 3.

                # Directly insert forces into RHS vector
                RHS0[allDofMem[0]] += fx
                RHS0[allDofMem[1]] += fy
                RHS0[allDofMem[2]] += fz
                RHS0[allDofMem[3]] += fx
                RHS0[allDofMem[4]] += fy
                RHS0[allDofMem[5]] += fz
                RHS0[allDofMem[6]] += fx
                RHS0[allDofMem[7]] += fy
                RHS0[allDofMem[8]] += fz

    if dim == 2:

        # Compute nodal load contributions (fX, fY)
        if loadedBCNodes[0, 0] != 0:

            # loop through nodal loads
            for el in range(np.size(loadedBCNodes, 0)):

                # constant load in x-direction
                if loadedBCNodes[el, 0] == 1:
                    dof1 = 2 * (int(loadedBCNodes[el, 1]) + 1) - 2
                    RHS[dof1] += loadStep * loadedBCNodes[el, 2]

                # constant load in y-direction
                elif loadedBCNodes[el, 0] == 2:
                    dof1 = 2 * (int(loadedBCNodes[el, 1]) + 1) - 1
                    RHS[dof1] += loadStep * loadedBCNodes[el, 2]

        # Compute edge load contributions (edgeX, edgeY)
        if loadedBCEdges[0, 0] != 0:

            # loop through nodal loads
            for el in range(np.size(loadedBCEdges, 0)):
                # TODO: constant edge length?
                # TODO: membrane thickness vs. cable thickness for integration?

                if loadedBCEdges[el, 0] == 1: # normalX

                    dof1 = 2 * int(loadedBCEdges[el, 1] + 1) - 2
                    dof2 = 2 * int(loadedBCEdges[el, 2] + 1) - 2

                    dy = fabs(Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])])

                    RHS[dof1] += loadStep * dy * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += loadStep * dy * loadedBCEdges[el, 3] / 2

                elif loadedBCEdges[el, 0] == 2: # normalY

                    dof1 = 2 * int(loadedBCEdges[el, 1] + 1) - 1
                    dof2 = 2 * int(loadedBCEdges[el, 2] + 1) - 1

                    dx = fabs(X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])])

                    RHS[dof1] += loadStep * dx * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += loadStep * dx * loadedBCEdges[el, 3] / 2

                elif loadedBCEdges[el, 0] == 4: # shearX

                    dof1 = 2 * int(loadedBCEdges[el, 1] + 1) - 2
                    dof2 = 2 * int(loadedBCEdges[el, 2] + 1) - 2

                    dx = fabs(X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])])

                    RHS[dof1] += loadStep * dx * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += loadStep * dx * loadedBCEdges[el, 3] / 2

                elif loadedBCEdges[el, 0] == 5: # shearY

                    dof1 = 2 * int(loadedBCEdges[el, 1] + 1) - 1
                    dof2 = 2 * int(loadedBCEdges[el, 2] + 1) - 1

                    dy = fabs(Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])])

                    RHS[dof1] += loadStep * dy * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += loadStep * dy * loadedBCEdges[el, 3] / 2

                # # constant load in x and y-direction with Sx, Sy magnitude (FSI in BC definition)
                # elif loadedBCEdges[el, 0] == 4:
                #
                #     # x-direction
                #     dof1 = 2 * int(loadedBCEdges[el, 1] + 1) - 2
                #     dof2 = 2 * int(loadedBCEdges[el, 2] + 1) - 2
                #
                #     dy = fabs(Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])])
                #
                #     RHS[dof1] += loadStep * dy * Sx[el] / 2
                #     RHS[dof2] += loadStep * dy * Sx[el] / 2
                #
                #     # y-direction
                #     dof1 = 2 * int(loadedBCEdges[el, 1] + 1) - 1
                #     dof2 = 2 * int(loadedBCEdges[el, 2] + 1) - 1
                #
                #     dx = fabs(X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])])
                #
                #     RHS[dof1] += loadStep * dx * Sy[el] / 2
                #     RHS[dof2] += loadStep * dx * Sy[el] / 2

        # cable elements
        if pre_active[0] == 1 or pre_active[1] == 1:

            # pre-stress and pre-strain in cable
            for el in range(nelemsCable):

                # Find degrees of freedom from current element
                allDofCable[0] = 2 * (NCable[el, 1] + 1) - 2
                allDofCable[1] = 2 * (NCable[el, 1] + 1) - 1
                allDofCable[2] = 2 * (NCable[el, 2] + 1) - 2
                allDofCable[3] = 2 * (NCable[el, 2] + 1) - 1

                # f_ext = B.T * (E * pre_strain_cable - pre_stress_cable) * area * L
                # B = [-X_21, -Y_21, X_21, Y_21]) / L ** 2
                fx = area2[el] * (E2[el] * pre_strain_cable[el] - pre_stress_cable[el]) / L0[el]

                RHS[allDofCable[0]] += (X[NCable[el, 1]] - X[NCable[el, 2]]) * fx
                RHS[allDofCable[1]] += (Y[NCable[el, 1]] - Y[NCable[el, 2]]) * fx
                RHS[allDofCable[2]] += (X[NCable[el, 2]] - X[NCable[el, 1]]) * fx
                RHS[allDofCable[3]] += (Y[NCable[el, 2]] - Y[NCable[el, 1]]) * fx

            # print(np.asarray(RHS))
        #
        # # membrane elements
        # if np.sum(pre_u) != 0:
        #     print("in RHS.pyx change if statement to 'if pre_u_flag is True' in line 302...")
        #     pre_membrane_2d(RHS, NMem, X, Y, pre_u, J11Vec, J22Vec, J12Vec, area3, thetaVec,
        #                     nu, E3, t, pre_stress_membrane, pre_strain_membrane, nelemsMem)

    elif dim == 3:

        if loadedBCNodes[0, 0] != 0:

            # loop through nodal loads
            for el in range(np.size(loadedBCNodes, 0)):

                if loadedBCNodes[el, 0] == 11:
                    dof1 = 3 * int(loadedBCNodes[el, 1] + 1) - 3
                    RHS[dof1] += loadStep * loadedBCNodes[el, 2]

                elif loadedBCNodes[el, 0] == 12:
                    dof1 = 3 * int(loadedBCNodes[el, 1] + 1) - 2
                    RHS[dof1] += loadStep * loadedBCNodes[el, 2]

                elif loadedBCNodes[el, 0] == 13:
                    dof1 = 3 * int(loadedBCNodes[el, 1] + 1) - 1
                    RHS[dof1] += loadStep * loadedBCNodes[el, 2]

        # Compute edge load contributions (edgeX, edgeY, edgeZ, edgeNormal)
        if loadedBCEdges[0, 0] != 0:

            # loop through nodal loads
            for el in range(np.size(loadedBCEdges, 0)):

                if loadedBCEdges[el, 0] == 1: # edgeX

                    dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 3
                    dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 3

                    # projected length on y-z plane
                    crossX = sqrt((Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])]) *
                                  (Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])]) +
                                  (Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])]) *
                                  (Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])]))

                    RHS[dof1] += crossX * loadStep * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += crossX * loadStep * loadedBCEdges[el, 3] / 2

                elif loadedBCEdges[el, 0] == 2: # edgeY

                    dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 2
                    dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 2

                    # projected length on x-z plane
                    crossY = sqrt((X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])]) *
                                  (X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])]) +
                                  (Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])]) *
                                  (Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])]))

                    RHS[dof1] += crossY * loadStep * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += crossY * loadStep * loadedBCEdges[el, 3] / 2

                elif loadedBCEdges[el, 0] == 3: # edgeZ

                    dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 1
                    dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 1

                    # projected length on x-y plane
                    crossZ = sqrt((X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])]) *
                                  (X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])]) +
                                  (Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])]) *
                                  (Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])]))

                    RHS[dof1] += crossZ * loadStep * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += crossZ * loadStep * loadedBCEdges[el, 3] / 2

                elif loadedBCEdges[el, 0] == 4: # shearX

                    dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 3
                    dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 3

                    dx = fabs(X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])])

                    RHS[dof1] += loadStep * dx * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += loadStep * dx * loadedBCEdges[el, 3] / 2

                elif loadedBCEdges[el, 0] == 5: # shearY

                    dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 2
                    dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 2

                    dy = fabs(Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])])

                    RHS[dof1] += loadStep * dy * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += loadStep * dy * loadedBCEdges[el, 3] / 2

                elif loadedBCEdges[el, 0] == 6: # shearZ

                    dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 1
                    dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 1

                    dy = fabs(Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])])

                    RHS[dof1] += loadStep * dy * loadedBCEdges[el, 3] / 2
                    RHS[dof2] += loadStep * dy * loadedBCEdges[el, 3] / 2

                # # constant load in x, y and z-direction with Sx, Sy, Sz magnitude (FSI in BC definition)
                # elif loadedBCEdges[el, 0] == 4:
                #
                #     # x-direction
                #     dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 3
                #     dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 3
                #
                #     # projected length on y-z plane
                #     crossX = sqrt((Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])]) *
                #                   (Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])]) +
                #                   (Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])]) *
                #                   (Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])]))
                #
                #     # integrate Sx over cable
                #     RHS[dof1] += crossX * loadStep * Sx[el] / 2
                #     RHS[dof2] += crossX * loadStep * Sx[el] / 2
                #
                #     # y-direction
                #     dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 2
                #     dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 2
                #
                #     # projected length on x-z plane
                #     crossY = sqrt((X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])]) *
                #                   (X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])]) +
                #                   (Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])]) *
                #                   (Z[int(loadedBCEdges[el, 2])] - Z[int(loadedBCEdges[el, 1])]))
                #
                #     # integrate Sx over cable
                #     RHS[dof1] += crossY * loadStep * Sy[el] / 2
                #     RHS[dof2] += crossY * loadStep * Sy[el] / 2
                #
                #     # z-direction
                #     dof1 = 3 * int(loadedBCEdges[el, 1] + 1) - 1
                #     dof2 = 3 * int(loadedBCEdges[el, 2] + 1) - 1
                #
                #     # projected length on x-y plane
                #     crossZ = sqrt((X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])]) *
                #                   (X[int(loadedBCEdges[el, 2])] - X[int(loadedBCEdges[el, 1])]) +
                #                   (Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])]) *
                #                   (Y[int(loadedBCEdges[el, 2])] - Y[int(loadedBCEdges[el, 1])]))
                #
                #     # integrate Sx over cable
                #     RHS[dof1] += crossZ * loadStep * Sz[el] / 2
                #     RHS[dof2] += crossZ * loadStep * Sz[el] / 2

        # pre-stress and pre-strain in cable
        for el in range(nelemsCable):

            # Find degrees of freedom from current element
            allDofCable[0] = 3 * (NCable[el, 1] + 1) - 3
            allDofCable[1] = 3 * (NCable[el, 1] + 1) - 2
            allDofCable[2] = 3 * (NCable[el, 1] + 1) - 1
            allDofCable[3] = 3 * (NCable[el, 2] + 1) - 3
            allDofCable[4] = 3 * (NCable[el, 2] + 1) - 2
            allDofCable[5] = 3 * (NCable[el, 2] + 1) - 1

            # f_ext = B.T * (E * pre_strain_cable - pre_stress_cable) * area * L
            # B = [-X_21, -Y_21, -Z_21, X_21, Y_21, Z_21]) / L ** 2
            fx = area2[el] * (E2[el] * pre_strain_cable[el] - pre_stress_cable[el]) / L0[el]

            RHS[allDofCable[0]] += (X[NCable[el, 1]] - X[NCable[el, 2]]) * fx
            RHS[allDofCable[1]] += (Y[NCable[el, 1]] - Y[NCable[el, 2]]) * fx
            RHS[allDofCable[2]] += (Z[NCable[el, 1]] - Z[NCable[el, 2]]) * fx
            RHS[allDofCable[3]] += (X[NCable[el, 2]] - X[NCable[el, 1]]) * fx
            RHS[allDofCable[4]] += (Y[NCable[el, 2]] - Y[NCable[el, 1]]) * fx
            RHS[allDofCable[5]] += (Z[NCable[el, 2]] - Z[NCable[el, 1]]) * fx

        # Loop through all pressurised elements
        for el in range(nPressurised):

            # Find degrees of freedom from current element
            allDofMem[0] = 3 * (NMem[elPressurised[el], 1] + 1) - 3
            allDofMem[1] = 3 * (NMem[elPressurised[el], 1] + 1) - 2
            allDofMem[2] = 3 * (NMem[elPressurised[el], 1] + 1) - 1
            allDofMem[3] = 3 * (NMem[elPressurised[el], 2] + 1) - 3
            allDofMem[4] = 3 * (NMem[elPressurised[el], 2] + 1) - 2
            allDofMem[5] = 3 * (NMem[elPressurised[el], 2] + 1) - 1
            allDofMem[6] = 3 * (NMem[elPressurised[el], 3] + 1) - 3
            allDofMem[7] = 3 * (NMem[elPressurised[el], 3] + 1) - 2
            allDofMem[8] = 3 * (NMem[elPressurised[el], 3] + 1) - 1

            # cross product components of current CST element
            crossX = (Y[NMem[elPressurised[el], 2]] - Y[NMem[elPressurised[el], 1]]) * \
                     (Z[NMem[elPressurised[el], 3]] - Z[NMem[elPressurised[el], 1]]) - \
                     (Y[NMem[elPressurised[el], 3]] - Y[NMem[elPressurised[el], 1]]) * \
                     (Z[NMem[elPressurised[el], 2]] - Z[NMem[elPressurised[el], 1]])

            crossY = (X[NMem[elPressurised[el], 3]] - X[NMem[elPressurised[el], 1]]) * \
                     (Z[NMem[elPressurised[el], 2]] - Z[NMem[elPressurised[el], 1]]) - \
                     (X[NMem[elPressurised[el], 2]] - X[NMem[elPressurised[el], 1]]) * \
                     (Z[NMem[elPressurised[el], 3]] - Z[NMem[elPressurised[el], 1]])

            crossZ = (X[NMem[elPressurised[el], 2]] - X[NMem[elPressurised[el], 1]]) * \
                     (Y[NMem[elPressurised[el], 3]] - Y[NMem[elPressurised[el], 1]]) - \
                     (X[NMem[elPressurised[el], 3]] - X[NMem[elPressurised[el], 1]]) * \
                     (Y[NMem[elPressurised[el], 2]] - Y[NMem[elPressurised[el], 1]])

            # force components (pressure = p/6 * cross, stressX = Sx * A / 3)
            fx = loadStep * crossX * p[el] / 6
            fy = loadStep * crossY * p[el] / 6
            fz = loadStep * crossZ * p[el] / 6

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

        # Loop through all pressurised elements
        for el in range(nFSI):

            # Find degrees of freedom from current element
            allDofMem[0] = 3 * (NMem[elFSI[el], 1] + 1) - 3
            allDofMem[1] = 3 * (NMem[elFSI[el], 1] + 1) - 2
            allDofMem[2] = 3 * (NMem[elFSI[el], 1] + 1) - 1
            allDofMem[3] = 3 * (NMem[elFSI[el], 2] + 1) - 3
            allDofMem[4] = 3 * (NMem[elFSI[el], 2] + 1) - 2
            allDofMem[5] = 3 * (NMem[elFSI[el], 2] + 1) - 1
            allDofMem[6] = 3 * (NMem[elFSI[el], 3] + 1) - 3
            allDofMem[7] = 3 * (NMem[elFSI[el], 3] + 1) - 2
            allDofMem[8] = 3 * (NMem[elFSI[el], 3] + 1) - 1

            # cross product components of current CST element
            crossX = (Y[NMem[elFSI[el], 2]] - Y[NMem[elFSI[el], 1]]) * \
                     (Z[NMem[elFSI[el], 3]] - Z[NMem[elFSI[el], 1]]) - \
                     (Y[NMem[elFSI[el], 3]] - Y[NMem[elFSI[el], 1]]) * \
                     (Z[NMem[elFSI[el], 2]] - Z[NMem[elFSI[el], 1]])

            crossY = (X[NMem[elFSI[el], 3]] - X[NMem[elFSI[el], 1]]) * \
                     (Z[NMem[elFSI[el], 2]] - Z[NMem[elFSI[el], 1]]) - \
                     (X[NMem[elFSI[el], 2]] - X[NMem[elFSI[el], 1]]) * \
                     (Z[NMem[elFSI[el], 3]] - Z[NMem[elFSI[el], 1]])

            crossZ = (X[NMem[elFSI[el], 2]] - X[NMem[elFSI[el], 1]]) * \
                     (Y[NMem[elFSI[el], 3]] - Y[NMem[elFSI[el], 1]]) - \
                     (X[NMem[elFSI[el], 3]] - X[NMem[elFSI[el], 1]]) * \
                     (Y[NMem[elFSI[el], 2]] - Y[NMem[elFSI[el], 1]])

            # current area
            area = 0.5 * sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ)

            # force components stressX = Sx * A / 3 + pFSI * crossX / 6
            fx = loadStep * (crossX * pFSI[el] + 2 * Sx[el] * area) / 6
            fy = loadStep * (crossY * pFSI[el] + 2 * Sy[el] * area) / 6
            fz = loadStep * (crossZ * pFSI[el] + 2 * Sz[el] * area) / 6

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
    if gravity == 1:
        # print("RHS0 = {}".format(np.asarray(RHS0)))
        # print("RHS = {}".format(np.asarray(RHS)))
        add_vv(RHS, RHS0, RHS)
        # print("RHS + RHS0 = {}".format(np.asarray(RHS)[0:20]))

    add_vv(RHS, multiply_vs(force_vector, loadStep, force_vector), RHS)


cdef int pre_membrane_2d(double [:] RHS,
                         int [:, ::1] NMem,
                         double [:] X,
                         double [:] Y,
                         double [:] pre_u,
                         long double [:] J11Vec,
                         long double [:] J22Vec,
                         long double [:] J12Vec,
                         double [:] area3,
                         double [:] thetaVec,
                         double [:] nu,
                         double [:] E3,
                         double [:] t,
                         double [:, ::1] pre_stress_membrane,
                         double [:, ::1] pre_strain_membrane,
                         unsigned int nelemsMem) except -1:

    cdef:

        double Q11, Q21, Q31, Q22, Q23, Q33, X21, X31, Y21, Y31, C11, C12, C22, C33, s, c, s2, c2
        double area

        unsigned int [:] allDofMem = np.empty(6, dtype=np.uintc)

        unsigned int el

    # pre-stress and pre-strain in membrane
    for el in range(nelemsMem):

        Q11 = 1 / (J11Vec[el] * J11Vec[el])
        Q21 = (J12Vec[el] * J12Vec[el]) / (J11Vec[el] * J11Vec[el] * J22Vec[el] * J22Vec[el])
        Q31 = - 2 * J12Vec[el] / (J11Vec[el] * J11Vec[el] * J22Vec[el])
        Q22 = 1 / (J22Vec[el] * J22Vec[el])
        Q23 = - J12Vec[el] / (J11Vec[el] * J22Vec[el] * J22Vec[el])
        Q33 = 1 / (J11Vec[el] * J22Vec[el])

        X21 = X[NMem[el, 2]] - X[NMem[el, 1]]
        X31 = X[NMem[el, 3]] - X[NMem[el, 1]]

        Y21 = Y[NMem[el, 2]] - Y[NMem[el, 1]]
        Y31 = Y[NMem[el, 3]] - Y[NMem[el, 1]]

        # local constitutive matrix parameters
        C11 = E3[el] / (1 - nu[el] * nu[el])
        C12 = nu[el] * E3[el] / (1 - nu[el] * nu[el])
        C22 = E3[el] / (1 - nu[el] * nu[el])
        C33 = 0.5 * (1 - nu[el]) * E3[el] / (1 - nu[el] * nu[el])

        # transformation between local and global coordinate system
        s = sin(thetaVec[el])
        c = cos(thetaVec[el])
        s2 = s * s
        c2 = c * c

        area = area3[el]

        # Find degrees of freedom from current element
        allDofMem[0] = 2 * (NMem[el, 1] + 1) - 2
        allDofMem[1] = 2 * (NMem[el, 1] + 1) - 1
        allDofMem[2] = 2 * (NMem[el, 2] + 1) - 2
        allDofMem[3] = 2 * (NMem[el, 2] + 1) - 1
        allDofMem[4] = 2 * (NMem[el, 3] + 1) - 2
        allDofMem[5] = 2 * (NMem[el, 3] + 1) - 1
        # print(thetaVec[el] * 180. / np.pi)
        # print(np.asarray(pre_stress_membrane))
        # print(np.asarray(pre_strain_membrane))
        # print(np.asarray(pre_u))
        # f_ext = B_xy.T * (C * (pre_strain_membrane + Q * B_curv * u_pre) - pre_stress_membrane) * area * t
        # B_xy = T_sigma.T * Q * B_curv

        # pre_u ONLY
        RHS[allDofMem[0]] += area*t[el]*(pre_u[allDofMem[0]]*(-Q22*X31*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))) - X21*(C33*Q31*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C12*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))) + Q21*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s)))) + (-X21 - X31)*(C33*Q33*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[1]]*(-Q22*Y31*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))) - Y21*(C33*Q31*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C12*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))) + Q21*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s)))) + (-Y21 - Y31)*(C33*Q33*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[2]]*(X21*(C33*Q31*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C12*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))) + Q21*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s)))) + X31*(C33*Q33*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[3]]*(Y21*(C33*Q31*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C12*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))) + Q21*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s)))) + Y31*(C33*Q33*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[4]]*(Q22*X31*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))) + X21*(C33*Q33*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[5]]*(Q22*Y31*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))) + Y21*(C33*Q33*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s))))))
        RHS[allDofMem[1]] += area*t[el]*(pre_u[allDofMem[0]]*(-Q22*X31*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))) - X21*(C33*Q31*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C12*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))) + Q21*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s)))) + (-X21 - X31)*(C33*Q33*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[1]]*(-Q22*Y31*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))) - Y21*(C33*Q31*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C12*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))) + Q21*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s)))) + (-Y21 - Y31)*(C33*Q33*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[2]]*(X21*(C33*Q31*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C12*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))) + Q21*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s)))) + X31*(C33*Q33*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[3]]*(Y21*(C33*Q31*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C12*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))) + Q21*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s)))) + Y31*(C33*Q33*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[4]]*(Q22*X31*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))) + X21*(C33*Q33*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[5]]*(Q22*Y31*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))) + Y21*(C33*Q33*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + C22*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s))))))
        RHS[allDofMem[2]] += area*t[el]*(pre_u[allDofMem[0]]*(-Q22*X31*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))) - X21*(C33*Q31*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C12*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))) + Q21*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s)))) + (-X21 - X31)*(C33*Q33*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[1]]*(-Q22*Y31*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))) - Y21*(C33*Q31*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C12*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))) + Q21*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s)))) + (-Y21 - Y31)*(C33*Q33*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[2]]*(X21*(C33*Q31*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C12*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))) + Q21*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s)))) + X31*(C33*Q33*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[3]]*(Y21*(C33*Q31*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C12*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))) + Q21*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s)))) + Y31*(C33*Q33*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[4]]*(Q22*X31*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))) + X21*(C33*Q33*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[5]]*(Q22*Y31*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))) + Y21*(C33*Q33*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s)) + C22*(X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))))))
        RHS[allDofMem[3]] += area*t[el]*(pre_u[allDofMem[0]]*(-Q22*X31*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))) - X21*(C33*Q31*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C12*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s)))) + (-X21 - X31)*(C33*Q33*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[1]]*(-Q22*Y31*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))) - Y21*(C33*Q31*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C12*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s)))) + (-Y21 - Y31)*(C33*Q33*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[2]]*(X21*(C33*Q31*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C12*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s)))) + X31*(C33*Q33*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[3]]*(Y21*(C33*Q31*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C12*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s)))) + Y31*(C33*Q33*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[4]]*(Q22*X31*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))) + X21*(C33*Q33*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[5]]*(Q22*Y31*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))) + Y21*(C33*Q33*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s)) + C22*(Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))))))
        RHS[allDofMem[4]] += area*t[el]*(pre_u[allDofMem[0]]*(-Q22*X31*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))) - X21*(C33*Q31*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C12*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s)))) + (-X21 - X31)*(C33*Q33*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[1]]*(-Q22*Y31*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))) - Y21*(C33*Q31*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C12*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s)))) + (-Y21 - Y31)*(C33*Q33*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[2]]*(X21*(C33*Q31*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C12*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s)))) + X31*(C33*Q33*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[3]]*(Y21*(C33*Q31*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C12*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s)))) + Y31*(C33*Q33*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[4]]*(Q22*X31*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))) + X21*(C33*Q33*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[5]]*(Q22*Y31*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))) + Y21*(C33*Q33*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s)) + C22*(Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))))))
        RHS[allDofMem[5]] += area*t[el]*(pre_u[allDofMem[0]]*(-Q22*X31*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))) - X21*(C33*Q31*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C12*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s)))) + (-X21 - X31)*(C33*Q33*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[1]]*(-Q22*Y31*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))) - Y21*(C33*Q31*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C12*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s)))) + (-Y21 - Y31)*(C33*Q33*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[2]]*(X21*(C33*Q31*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C12*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s)))) + X31*(C33*Q33*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[3]]*(Y21*(C33*Q31*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q11*(C11*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C12*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))) + Q21*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s)))) + Y31*(C33*Q33*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[4]]*(Q22*X31*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))) + X21*(C33*Q33*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))))) + pre_u[allDofMem[5]]*(Q22*Y31*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))) + Y21*(C33*Q33*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + Q23*(C12*(Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s)) + C22*(Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))))))

        # RHS[allDofMem[0]] += area*t[el]*((C33*(Q33*X21*u[allDofMem[4]] + Q33*Y21*u[allDofMem[5]] + pre_strain_membrane[el, 2] + u[allDofMem[0]]*(-Q31*X21 + Q33*(-X21 - X31)) + u[allDofMem[1]]*(-Q31*Y21 + Q33*(-Y21 - Y31)) + u[allDofMem[2]]*(Q31*X21 + Q33*X31) + u[allDofMem[3]]*(Q31*Y21 + Q33*Y31)) - pre_stress_membrane[el, 2])*(-2*Q22*X31*c*s - X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))) + (C11*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C12*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_strain_membrane[el, 0])*(-Q22*X31*s2 - X21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-X21 - X31)*(Q23*s2 + Q33*c*s)) + (C12*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C22*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_stress_membrane[el, 1])*(-Q22*X31*c2 - X21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-X21 - X31)*(Q23*c2 - Q33*c*s)))
        #
        # RHS[allDofMem[1]] += area*t[el]*((C33*(Q33*X21*u[allDofMem[4]] + Q33*Y21*u[allDofMem[5]] + pre_strain_membrane[el, 2] + u[allDofMem[0]]*(-Q31*X21 + Q33*(-X21 - X31)) + u[allDofMem[1]]*(-Q31*Y21 + Q33*(-Y21 - Y31)) + u[allDofMem[2]]*(Q31*X21 + Q33*X31) + u[allDofMem[3]]*(Q31*Y21 + Q33*Y31)) - pre_stress_membrane[el, 2])*(-2*Q22*Y31*c*s - Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))) + (C11*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C12*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_strain_membrane[el, 0])*(-Q22*Y31*s2 - Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + (C12*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C22*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_stress_membrane[el, 1])*(-Q22*Y31*c2 - Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + (-Y21 - Y31)*(Q23*c2 - Q33*c*s)))
        #
        # RHS[allDofMem[2]] += area*t[el]*((C33*(Q33*X21*u[allDofMem[4]] + Q33*Y21*u[allDofMem[5]] + pre_strain_membrane[el, 2] + u[allDofMem[0]]*(-Q31*X21 + Q33*(-X21 - X31)) + u[allDofMem[1]]*(-Q31*Y21 + Q33*(-Y21 - Y31)) + u[allDofMem[2]]*(Q31*X21 + Q33*X31) + u[allDofMem[3]]*(Q31*Y21 + Q33*Y31)) - pre_stress_membrane[el, 2])*(X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + X31*(2*Q23*c*s + Q33*(c2 - s2))) + (X21*(Q11*c2 + Q21*s2 + Q31*c*s) + X31*(Q23*s2 + Q33*c*s))*(C11*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C12*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_strain_membrane[el, 0]) + (X21*(Q11*s2 + Q21*c2 - Q31*c*s) + X31*(Q23*c2 - Q33*c*s))*(C12*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C22*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_stress_membrane[el, 1]))
        #
        # RHS[allDofMem[3]] += area*t[el]*((C33*(Q33*X21*u[allDofMem[4]] + Q33*Y21*u[allDofMem[5]] + pre_strain_membrane[el, 2] + u[allDofMem[0]]*(-Q31*X21 + Q33*(-X21 - X31)) + u[allDofMem[1]]*(-Q31*Y21 + Q33*(-Y21 - Y31)) + u[allDofMem[2]]*(Q31*X21 + Q33*X31) + u[allDofMem[3]]*(Q31*Y21 + Q33*Y31)) - pre_stress_membrane[el, 2])*(Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) + Y31*(2*Q23*c*s + Q33*(c2 - s2))) + (Y21*(Q11*c2 + Q21*s2 + Q31*c*s) + Y31*(Q23*s2 + Q33*c*s))*(C11*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C12*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_strain_membrane[el, 0]) + (Y21*(Q11*s2 + Q21*c2 - Q31*c*s) + Y31*(Q23*c2 - Q33*c*s))*(C12*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C22*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_stress_membrane[el, 1]))
        #
        # RHS[allDofMem[4]] += area*t[el]*((C33*(Q33*X21*u[allDofMem[4]] + Q33*Y21*u[allDofMem[5]] + pre_strain_membrane[el, 2] + u[allDofMem[0]]*(-Q31*X21 + Q33*(-X21 - X31)) + u[allDofMem[1]]*(-Q31*Y21 + Q33*(-Y21 - Y31)) + u[allDofMem[2]]*(Q31*X21 + Q33*X31) + u[allDofMem[3]]*(Q31*Y21 + Q33*Y31)) - pre_stress_membrane[el, 2])*(2*Q22*X31*c*s + X21*(2*Q23*c*s + Q33*(c2 - s2))) + (Q22*X31*c2 + X21*(Q23*c2 - Q33*c*s))*(C12*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C22*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_stress_membrane[el, 1]) + (Q22*X31*s2 + X21*(Q23*s2 + Q33*c*s))*(C11*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C12*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_strain_membrane[el, 0]))
        #
        # RHS[allDofMem[5]] += area*t[el]*((C33*(Q33*X21*u[allDofMem[4]] + Q33*Y21*u[allDofMem[5]] + pre_strain_membrane[el, 2] + u[allDofMem[0]]*(-Q31*X21 + Q33*(-X21 - X31)) + u[allDofMem[1]]*(-Q31*Y21 + Q33*(-Y21 - Y31)) + u[allDofMem[2]]*(Q31*X21 + Q33*X31) + u[allDofMem[3]]*(Q31*Y21 + Q33*Y31)) - pre_stress_membrane[el, 2])*(2*Q22*Y31*c*s + Y21*(2*Q23*c*s + Q33*(c2 - s2))) + (Q22*Y31*c2 + Y21*(Q23*c2 - Q33*c*s))*(C12*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C22*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_stress_membrane[el, 1]) + (Q22*Y31*s2 + Y21*(Q23*s2 + Q33*c*s))*(C11*(-Q11*X21*u[allDofMem[0]] + Q11*X21*u[allDofMem[2]] - Q11*Y21*u[allDofMem[1]] + Q11*Y21*u[allDofMem[3]] + pre_strain_membrane[el, 0]) + C12*(pre_strain_membrane[el, 1] + u[allDofMem[0]]*(-Q21*X21 - Q22*X31 + Q23*(-X21 - X31)) + u[allDofMem[1]]*(-Q21*Y21 - Q22*Y31 + Q23*(-Y21 - Y31)) + u[allDofMem[2]]*(Q21*X21 + Q23*X31) + u[allDofMem[3]]*(Q21*Y21 + Q23*Y31) + u[allDofMem[4]]*(Q22*X31 + Q23*X21) + u[allDofMem[5]]*(Q22*Y31 + Q23*Y21)) - pre_strain_membrane[el, 0]))

        # RHS[allDofMem[0]] += area*t[el]*(pre_stress_membrane[el, 0]*(Q22*X31*s2 + X21*(Q11*c2 + Q21*s2 + Q31*c*s) - (-X21 - X31)*(Q23*s2 + Q33*c*s)) + pre_stress_membrane[el, 1]*(Q22*X31*c2 + X21*(Q11*s2 + Q21*c2 - Q31*c*s) - (-X21 - X31)*(Q23*c2 - Q33*c*s)) + pre_stress_membrane[el, 2]*(2*Q22*X31*c*s + X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) - (-X21 - X31)*(2*Q23*c*s + Q33*(c2 - s2))))
        #
        # RHS[allDofMem[1]] += area*t[el]*(pre_stress_membrane[el, 0]*(Q22*Y31*s2 + Y21*(Q11*c2 + Q21*s2 + Q31*c*s) - (-Y21 - Y31)*(Q23*s2 + Q33*c*s)) + pre_stress_membrane[el, 1]*(Q22*Y31*c2 + Y21*(Q11*s2 + Q21*c2 - Q31*c*s) - (-Y21 - Y31)*(Q23*c2 - Q33*c*s)) + pre_stress_membrane[el, 2]*(2*Q22*Y31*c*s + Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) - (-Y21 - Y31)*(2*Q23*c*s + Q33*(c2 - s2))))
        #
        # RHS[allDofMem[2]] += area*t[el]*(pre_stress_membrane[el, 0]*(-X21*(Q11*c2 + Q21*s2 + Q31*c*s) - X31*(Q23*s2 + Q33*c*s)) + pre_stress_membrane[el, 1]*(-X21*(Q11*s2 + Q21*c2 - Q31*c*s) - X31*(Q23*c2 - Q33*c*s)) + pre_stress_membrane[el, 2]*(-X21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) - X31*(2*Q23*c*s + Q33*(c2 - s2))))
        #
        # RHS[allDofMem[3]] += area*t[el]*(pre_stress_membrane[el, 0]*(-Y21*(Q11*c2 + Q21*s2 + Q31*c*s) - Y31*(Q23*s2 + Q33*c*s)) + pre_stress_membrane[el, 1]*(-Y21*(Q11*s2 + Q21*c2 - Q31*c*s) - Y31*(Q23*c2 - Q33*c*s)) + pre_stress_membrane[el, 2]*(-Y21*(-2*Q11*c*s + 2*Q21*c*s + Q31*(c2 - s2)) - Y31*(2*Q23*c*s + Q33*(c2 - s2))))
        #
        # RHS[allDofMem[4]] += area*t[el]*(pre_stress_membrane[el, 0]*(-Q22*X31*s2 - X21*(Q23*s2 + Q33*c*s)) + pre_stress_membrane[el, 1]*(-Q22*X31*c2 - X21*(Q23*c2 - Q33*c*s)) + pre_stress_membrane[el, 2]*(-2*Q22*X31*c*s - X21*(2*Q23*c*s + Q33*(c2 - s2))))
        #
        # RHS[allDofMem[5]] += area*t[el]*(pre_stress_membrane[el, 0]*(-Q22*Y31*s2 - Y21*(Q23*s2 + Q33*c*s)) + pre_stress_membrane[el, 1]*(-Q22*Y31*c2 - Y21*(Q23*c2 - Q33*c*s)) + pre_stress_membrane[el, 2]*(-2*Q22*Y31*c*s - Y21*(2*Q23*c*s + Q33*(c2 - s2))))