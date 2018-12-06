import numpy as np
cimport numpy as np
cimport cython

from scipy.sparse import spdiags

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef initialiseDirichletBC(unsigned int [:] fixedBCNodes,
                           unsigned int dim):

    # TODO: Exception of no dirichlet BC is defined

    # Count total number of fixed dofs and their node number
    cdef np.ndarray[unsigned int] npFixedBC = np.asarray(fixedBCNodes)
    cdef unsigned int i, length

    # 2D or 3D
    if dim == 2:

        indexNodeAll = np.where(npFixedBC[::2] == 0)[0]

        indexNodeX = np.where(npFixedBC[::2] == 1)[0]

        indexNodeY = np.where(npFixedBC[::2] == 2)[0]

        tempAll = npFixedBC[2 * (indexNodeAll + 1) - 1]
        tempX = npFixedBC[2 * (indexNodeX + 1) - 1]
        tempY = npFixedBC[2 * (indexNodeY + 1) - 1]

        fixX = 2 * (np.unique(np.append(tempAll, tempX)) + 1) - 2
        fixY = 2 * (np.unique(np.append(tempAll, tempY)) + 1) - 1

        length = len(fixX) + len(fixY)
        if length == 0:
            raise Warning("No Dirichlet boundary conditions found.")

        dofFixed = np.sort(np.hstack((fixX, fixY)))

    elif dim == 3:

        indexNodeAll = np.where(npFixedBC[::2] == 0)[0]

        indexNodeX = np.where(npFixedBC[::2] == 1)[0]

        indexNodeY = np.where(npFixedBC[::2] == 2)[0]

        indexNodeZ = np.where(npFixedBC[::2] == 3)[0]

        tempAll = npFixedBC[2 * (indexNodeAll + 1) - 1]
        tempX = npFixedBC[2 * (indexNodeX + 1) - 1]
        tempY = npFixedBC[2 * (indexNodeY + 1) - 1]
        tempZ = npFixedBC[2 * (indexNodeZ + 1) - 1]

        fixX = 3 * (np.unique(np.append(tempAll, tempX)) + 1) - 3
        fixY = 3 * (np.unique(np.append(tempAll, tempY)) + 1) - 2
        fixZ = 3 * (np.unique(np.append(tempAll, tempZ)) + 1) - 1

        length = len(fixX) + len(fixY) + len(fixZ)
        if length == 0:
            raise Warning("No Dirichlet boundary conditions found.")

        dofFixed = np.sort(np.hstack((fixX, fixY, fixZ)))

    # # Follower loads - which elements get load stiffness matrix KL
    # if self.followerLoad is True:
    #
    #     temp = []
    #
    #     # pressure on edge
    #     ind = np.where(self.loadedEdges[:, 0] == 5)[0]
    #
    #     # loop through edges and find nodes and corresponding element they belong to
    #     for i, edge in enumerate(ind):
    #
    #         # nodal indices of edge (already defined counter-clockwise)
    #         indEdge = self.loadedEdges[edge, 1:]
    #
    #         # edge in element
    #         logicID = np.reshape(np.in1d(self.N[:, 1:], indEdge), (np.size(self.N, 0), 3))
    #         elementID = np.where(np.sum(logicID, axis=1) == 2)
    #
    #         temp = np.append(temp, int(elementID[0]))
    #         temp = np.append(temp, indEdge)
    #
    #         if np.array_equal(logicID[elementID][0], np.array([True, True, False])):
    #             temp = np.append(temp, 0)
    #         elif np.array_equal(logicID[elementID][0], np.array([False, True, True])):
    #             temp = np.append(temp, 1)
    #         elif np.array_equal(logicID[elementID][0], np.array([True, False, True])):
    #             temp = np.append(temp, 2)
    #
    #         temp = np.append(temp, int(i))
    #
    #     temp = np.asarray(temp, dtype=int)
    #
    #     # FLelems:
    #     # 0: element ID
    #     # 1: nodes 1
    #     # 2: nodes 2
    #     # 3: edge ID (0 = [1, 2], 1 = [2, 3], 2 = [3, 1])
    #     # 4: loadedEdges ID
    #     self.FLelems = np.reshape(temp, (len(ind), 5))

    return dofFixed


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void initialiseLoadBC(int [:, ::1] N,
                           unsigned int [:] loadedBCNodes,
                           unsigned int dim,
                           unsigned int [:] dofLoaded,
                           unsigned int [:] dofLoadtype):

    # Count total number of fixed dofs and their node number
    cdef np.ndarray[int] npLoadBC = np.asarray(loadedBCNodes)

    # 2D or 3D
    if dim == 2:

        indexFX = np.where(npLoadBC[::2] == 1)[0]
        tempFX = npLoadBC[2 * (indexFX + 1) - 1]

        indexFY = np.where(npLoadBC[::2] == 2)[0]
        tempFY = npLoadBC[2 * (indexFY + 1) - 1]

        indexEdgeX = np.where(npLoadBC[::2] == 4)[0]
        tempEdgeX = npLoadBC[2 * (indexEdgeX + 1) - 1]

        indexEdgeY = np.where(npLoadBC[::2] == 5)[0]
        tempEdgeY = npLoadBC[2 * (indexEdgeY + 1) - 1]

        indexTraction = np.where(npLoadBC[::2] == 7)[0]
        tempTraction = npLoadBC[2 * (indexTraction + 1) - 1]

        indexFSI = np.where(npLoadBC[::2] == 8)[0]
        tempFSI = npLoadBC[2 * (indexFSI + 1) - 1]

        indexCpTop = np.where(npLoadBC[::2] == 9)[0]
        tempCpTop = npLoadBC[2 * (indexCpTop + 1) - 1]

        indexCpBot = np.where(npLoadBC[::2] == 10)[0]
        tempCpBot = npLoadBC[2 * (indexCpBot + 1) - 1]

        fixX = 3 * (np.unique(np.append(tempFX, tempEdgeX)) + 1) - 3
        fixY = 3 * (np.unique(np.append(tempFY, tempEdgeY)) + 1) - 2


    elif dim == 3:

        indexFX = np.where(npLoadBC[::2] == 1)[0]
        tempFX = npLoadBC[2 * (indexFX + 1) - 1]

        indexFY = np.where(npLoadBC[::2] == 2)[0]
        tempFY = npLoadBC[2 * (indexFY + 1) - 1]

        indexFZ = np.where(npLoadBC[::2] == 3)[0]
        tempFZ = npLoadBC[2 * (indexFZ + 1) - 1]

        indexEdgeX = np.where(npLoadBC[::2] == 4)[0]
        tempEdgeX = npLoadBC[2 * (indexEdgeX + 1) - 1]

        indexEdgeY = np.where(npLoadBC[::2] == 5)[0]
        tempEdgeY = npLoadBC[2 * (indexEdgeY + 1) - 1]

        indexEdgeZ = np.where(npLoadBC[::2] == 6)[0]
        tempEdgeZ = npLoadBC[2 * (indexEdgeZ + 1) - 1]

        indexTraction = np.where(npLoadBC[::2] == 7)[0]
        tempTraction = npLoadBC[2 * (indexTraction + 1) - 1]

        indexFSI = np.where(npLoadBC[::2] == 8)[0]
        tempFSI = npLoadBC[2 * (indexFSI + 1) - 1]

        indexCpTop = np.where(npLoadBC[::2] == 9)[0]
        tempCpTop = npLoadBC[2 * (indexCpTop + 1) - 1]

        indexCpBot = np.where(npLoadBC[::2] == 10)[0]
        tempCpBot = npLoadBC[2 * (indexCpBot + 1) - 1]

        fixX = 3 * (np.unique(np.append(tempFX, tempEdgeX)) + 1) - 3
        fixY = 3 * (np.unique(np.append(tempFY, tempEdgeY)) + 1) - 2
        fixZ = 3 * (np.unique(np.append(tempFZ, tempEdgeZ)) + 1) - 1

        dofFixed = np.sort(np.hstack((fixX, fixY, fixZ)))
        dofLoaded = np.empty(1, dtype=np.intc)
        dofLoadtype = np.empty(1, dtype=np.intc)

        print("test all loads for 3D in initialiseLoadBC")
        quit(1)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void correctBC3D(double [:] R,
                      int [:] dofFixed):

    cdef unsigned int i

    for i in dofFixed:

        R[i] = 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef object dirichlet_zero_matrix_modification(object matrix,
                                               int [:] dirichlet_zero_dofs):
    """Enforces dirichlet zero B.C.'s by zeroing-out rows and columns 
    associated with dirichlet dofs, and putting 1 on the diagonal there."""

    # cdef unsigned int zdof = np.asarray(dirichlet_zero_dofs)
    # cdef unsigned int N = matrix.shape[1]
    # cdef double [:] chi_interior = np.ones(N, dtype=np.double)
    # cdef double [:] chi_boundary = np.zeros(N, dtype=np.double)
    zdof = np.asarray(dirichlet_zero_dofs)
    N = matrix.shape[1]
    chi_interior = np.ones(N)
    chi_boundary = np.zeros(N)

    chi_interior[zdof] = 0.0
    I_interior = spdiags(chi_interior, [0], N, N).tocsc()

    chi_boundary = np.zeros(N)
    chi_boundary[zdof] = 1.0
    I_boundary = spdiags(chi_boundary, [0], N, N).tocsc()

    matrix_modified = I_interior * matrix * I_interior + I_boundary

    matrix_modified.eliminate_zeros()

    return matrix_modified