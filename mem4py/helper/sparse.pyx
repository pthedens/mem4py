# cython: language_level=3
# cython: boundcheck=False
import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse import coo_matrix

from ceygen.ceygenMath cimport multiply_vs, power_vs


# cdef extern from "math.h":
#     double sqrt(double m)
#     double fabs(double m)
#     double fmax(double m, double n)
from libc.math cimport sqrt, fabs, fmax
from libcpp.vector cimport vector


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef object sparsityPattern(int [:, ::1] NCable,
                            int [:, ::1] NMem,
                            unsigned int nelemsCable,
                            unsigned int nelemsMem,
                            unsigned int ndof,
                            unsigned int dim):
    """
    determines non-zero entry locations in stiffness matrix and creates map from coordinate (coo_matrix)
    to compressed sparse row (csr_matrix) in vector form, sorted by rows..
    :param NCable: 
    :param NMem: 
    :param nelemsCable: 
    :param nelemsMem: 
    :param ndof: 
    :param dim: 
    :return: 
    """

    cdef:
        double [:] data = np.ones(dim * dim * (9 * nelemsMem + 4 * nelemsCable), dtype=np.double)

        unsigned int [:] row = np.empty(dim * dim * (9 * nelemsMem + 4 * nelemsCable), dtype=np.uintc)
        unsigned int [:] col = np.empty(dim * dim * (9 * nelemsMem + 4 * nelemsCable), dtype=np.uintc)

        unsigned int [:] allDofCable = np.empty(2 * dim, dtype=np.uintc)
        unsigned int [:] allDofMem = np.empty(3 * dim, dtype=np.uintc)

        Py_ssize_t el, i, j, runner, index = 0

    # Loop over all cable elements
    for el in range(nelemsCable):

        # Find degrees of freedom from current element
        for i in range(dim):
            allDofCable[i]       = dim * (NCable[el, 1] + 1) - (dim - i % dim)
            allDofCable[i + dim] = dim * (NCable[el, 2] + 1) - (dim - i % dim)

        # row major flattening
        runner = 0
        for i in range(2 * dim):
            for j in range(2 * dim):
                row[index + runner] = allDofCable[i]
                col[index + runner] = allDofCable[j]
                runner += 1

        # increase index by length of entries in local stiffness matrix Kloc
        index += 4 * dim * dim

    # Loop over all membrane elements
    for el in range(nelemsMem):

        # Find degrees of freedom from current element
        for i in range(dim):
            allDofMem[i]           = dim * (NMem[el, 1] + 1) - (dim - i % dim)
            allDofMem[i + dim]     = dim * (NMem[el, 2] + 1) - (dim - i % dim)
            allDofMem[i + 2 * dim] = dim * (NMem[el, 3] + 1) - (dim - i % dim)

        # row major flattening
        runner = 0
        for i in range(3 * dim):
            for j in range(3 * dim):
                row[index + runner] = allDofMem[i]
                col[index + runner] = allDofMem[j]
                runner += 1

        # increase index by length of entries in local stiffness matrix Kloc
        index += 9 * dim * dim

    # assemble stiffness matrix
    K = coo_matrix((data, (row, col)), shape=(ndof, ndof)).tocsr()

    # extract index pointer from K
    indptr = K.indptr
    indices = K.indices

    # order K entries and find duplicates which are added in data vector
    order = np.lexsort((col, row))
    row2 = np.asarray(row)[order]
    col2 = np.asarray(col)[order]
    unique_mask = ((row2[1:] != row2[:-1]) |
                   (col2[1:] != col2[:-1]))
    unique_mask = np.append(True, unique_mask)
    duplicates = np.where(~unique_mask)[0]

    # order the order vector
    order2 = np.argsort(order)
    orderSorted = np.sort(order)

    cdef long long [:] orderSortedC = orderSorted
    count = 0
    ind = 0

    # run through orderSorted and find duplicate indices, and shift all above
    for i in range(len(order2)):
        if ind < len(duplicates):
            if i == duplicates[ind]:
                count -= 1
                ind += 1
        orderSortedC[i] += count

    # put back to element loop order
    order2 = np.asarray(orderSortedC)[order2]

    return order2, indptr, indices


@cython.profile(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int addRows(double [:] data,
                 unsigned int [:] indptr,
                 unsigned int ndof,
                 double [:] diagK,
                 double * alpha,
                 double [:] M,
                 double [:] Minv,
                 double [:] sumRowWithDiag,
                 unsigned int iteration,
                 unsigned int dim,
                 str method,
                 double lam) nogil:
    """
    estimate largest eigenvalue of stiffness matrix by adding absolute
    value of each row entry, see Alamatian el al. 2012
    :param data: 
    :param indptr: 
    :param ndof: 
    :param diagK: 
    :param alpha: 
    :param M: 
    :param Minv: 
    :param sumRowWithDiag: 
    :return: 
    """
    cdef:
        unsigned int zero_detected = 0
        double betaRow, alphaSqrt, noDiag, alphaSqrtOld = -1
        Py_ssize_t i, j, ind = 0

    # set sumRowWithDiag to zero
    sumRowWithDiag[...] = 0

    if method == "KDR1":
        multiply_vs(diagK, lam * 0.5, M)

    elif method == "Alamatian":

        # loop through columns of K and determine alpha
        for i in range(ndof):

            # sum abs of K columns
            for j in range(indptr[i + 1] - indptr[i]):
                sumRowWithDiag[i] += fabs(data[ind])
                ind += 1

            # # Rezaiee-Pajand et al. 2014
            # if fabs(diagK[i]) > 0:
            #     M[i] = 1.6 * fmax(diagK[i] / 2, sumRowWithDiag[i] / 4)

            ## ALAMATIAN 2012 METHOD START
            if diagK[i] > 0:

                noDiag = sumRowWithDiag[i] - fabs(diagK[i])
                betaRow = noDiag / diagK[i]

                # check for row condition
                if diagK[i] > noDiag:
                    alphaSqrt = (1 - sqrt(1 - betaRow * betaRow)) / betaRow
                elif diagK[i] <= 0.5 * noDiag or diagK[i] == noDiag:
                    alphaSqrt = (betaRow + 2 - 2 * sqrt(1 + betaRow)) / betaRow
                else:
                    alphaSqrt = betaRow - sqrt(betaRow * betaRow - 1)

                if alphaSqrt > alphaSqrtOld:
                    alphaSqrtOld = alphaSqrt

        alpha[0] = alphaSqrtOld * alphaSqrtOld
        alphaSqrt = (1 + alphaSqrtOld * alphaSqrtOld) / (2 * (1 + alphaSqrtOld) * (1 + alphaSqrtOld))

        multiply_vs(sumRowWithDiag, alphaSqrt, M)
        # ALAMATIAN 2012 METHOD ENDS HERE

    # Barnes method
    elif method == "Barnes":

        # loop through columns of K and determine alpha
        for i in range(ndof):

            # sum abs of K columns
            for j in range(indptr[i + 1] - indptr[i]):

                sumRowWithDiag[i] += fabs(data[ind])
                ind += 1

        multiply_vs(sumRowWithDiag, lam / 2, M)

    else:
        raise Exception("no method defined for DR."
                        "Choose either Barnes, Alamatian, or KDR1")

    for i in range(ndof):
        if M[i] == 0.:
            zero_detected = 1
            break

    if zero_detected == 1:

        if dim == 2:

            # Set zero mass entries to
            for i in range(ndof):

                # set mass to largest nodal value
                if fabs(M[i]) == 0:

                    if (i / 2) % 2:  # x
                        M[i] = M[i + 1]
                    else:  # y
                        M[i] = M[i - 1]

        elif dim == 3:

            ind = 0
            for i in range(ndof/3):
                alphaSqrt = fmax(M[ind], M[ind+1])
                alphaSqrt = fmax(alphaSqrt, M[ind+2])
                M[ind] = alphaSqrt
                M[ind+1] = alphaSqrt
                M[ind+2] = alphaSqrt
                ind += 3

    power_vs(M, -1, Minv)
