import numpy as np
cimport numpy as np
cimport cython
import pprint

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(True)  # turn off negative index wrapping for entire function
cdef int checkInput(object problem) except -1:

    if not isinstance(problem, dict):
        raise Exception("No problem dict found. Please provide problem dict")

    if not "msh" in problem:
        raise Exception("Define mesh file name msh")

    if not "mat" in problem:
        raise Exception("Define material model mat: SVIso or SVOrtho")

    if problem["mat"] == "SVIso":
        if not "EMem" in problem and "EBar" in problem:
            raise Exception("Define Young's modulus EMem and/or EBar")
        if not "nu" in problem and "EMem" in problem:
            raise Exception("Define Poisson's ratio nu")
        if not "t" in problem:
            raise Exception("Define thickness t")
    elif "mat" in problem == "SVOrtho":
        if not "E1Mem" in problem:
            raise Exception("Define Young's modulus E1")
        if not "E2Mem" in problem:
            raise Exception("Define Young's modulus E2")
        if not "nu12" in problem:
            raise Exception("Define Poisson's ratio nu12")
        if not "nu21" in problem:
            raise Exception("Define Poisson's ratio nu21")
        if not "t" in problem:
            raise Exception("Define thickness t")

    if "DR" in problem is True:
        if not problem["epsilonR"]:
            raise Exception("Define force convergence criterion epsilonR")
        if not problem["epsilonKE"]:
            raise Exception("Define kinetic energy convergence criterion epsilonKE")

    # if not problem["p"]:
    #     raise Exception("Define pressure magnitude p")

    print("Following input parameters defined:\n")
    pprint.pprint(problem)
    print("")