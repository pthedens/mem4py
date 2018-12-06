cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void writeVectorNode3D(double [:] u, unsigned int nnodes, str name, object fout):

    cdef unsigned int n

    fout.write('VECTORS %s float \n' % name)

    for n in range(0, nnodes):
        ux = u[3 * (n + 1) - 3]
        uy = u[3 * (n + 1) - 2]
        uz = u[3 * (n + 1) - 1]
        fout.write('%s %s %s\n' % (ux, uy, uz))

    fout.write("\n")

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void writeVectorNode2D(double [:] u, unsigned int nnodes, str name, object fout):

    cdef unsigned int n

    fout.write('VECTORS %s float \n' % name)

    for n in range(0, nnodes):
        ux = u[2 * (n + 1) - 2]
        uy = u[2 * (n + 1) - 1]
        fout.write('%s %s %s\n' % (ux, uy, 0))

    fout.write("\n")

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void writeVector2D(double [:] u, unsigned int nnodes, str name, object fout):

    cdef unsigned int n

    fout.write('VECTORS %s float \n' % name)

    for n in range(0, nnodes):
        ux = u[2 * (n + 1) - 2]
        uy = u[2 * (n + 1) - 1]
        uz = 0
        fout.write('%s %s %s\n' % (ux, uy, uz))

    fout.write("\n")

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void writeVectorElement(double [:, ::1] vector, str name, unsigned int nelems , object fout):

    cdef unsigned int n

    fout.write('VECTORS %s float \n' % name)

    for n in range(0, nelems):

        # 3x3 symmetric tensor
        fout.write('%s %s %s\n' % (vector[n, 0], vector[n, 1], vector[n, 2]))

    fout.write('\n')

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void writeScalarElement(double [:] scalar, str name, unsigned int nelems , object fout):

    cdef unsigned int n

    fout.write('SCALARS %s float 1\n' % name)
    fout.write('LOOKUP_TABLE default \n')

    for n in range(0, nelems):

        # 3x3 symmetric tensor
        fout.write('%s\n' % (scalar[n]))

    fout.write('\n')

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void writeScalarElementInt(unsigned int [:] scalar, str name, unsigned int nelems , object fout):

    cdef unsigned int n

    fout.write('SCALARS %s float 1\n' % name)
    fout.write('LOOKUP_TABLE default \n')

    for n in range(0, nelems):

        # 3x3 symmetric tensor
        fout.write('%s\n' % (scalar[n]))

    fout.write('\n')