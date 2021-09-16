# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
cimport cython

from elements.cable cimport cable2DFintAndK
from elements.cable cimport cable3DFintAndK
from elements.cable cimport cable2D_internal_force_vector
from elements.cable cimport cable3D_internal_force_vector
from elements.membrane cimport membrane2DStrain
from elements.membrane cimport membrane2DKmat
from elements.membrane cimport membrane3DStrain
from elements.membrane cimport membrane3DKmat
# from elements.membrane cimport membrane3DKmatVisc
from elements.membrane cimport membraneStress
from elements.membrane cimport membraneWrinklingJarasjarungkiat
from elements.membrane cimport membrane2D_internal_force_vector
from elements.membrane cimport membrane3D_internal_force_vector


cdef int computeuDotDot(double [:] Minv,
                        double [:] X,
                        double [:] Y,
                        double [:] Z,
                        int [:, ::1] NMem,
                        int [:, ::1] NCable,
                        long double [:] J11Vec,
                        long double [:] J22Vec,
                        long double [:] J12Vec,
                        unsigned int nelemsMem,
                        unsigned int nelemsCable,
                        unsigned int ndof,
                        double [:] E3,
                        double [:] E2,
                        double [:] nu,
                        double [:] t,
                        double [:] area2,
                        double alpha,
                        double beta,
                        double [:] thetaVec,
                        double [:] area3,
                        double [:] L0,
                        double [:] Fint,
                        double [:] RHS,
                        double [:, ::1] Ew,
                        double [:] p,
                        unsigned int [:] order,
                        unsigned int [:] indices,
                        unsigned int [:] indptr,
                        unsigned int [:] state,
                        double [:] data,
                        double [:] uDot,
                        double [:] uDotDot,
                        double [:] diagK,
                        unsigned int dim,
                        unsigned int wrinklingFlag,
                        double [:] P,
                        double [:] alpha_array,
                        unsigned int wrinkling_iter,
                        unsigned int iter_goal,
                        double sigma_max,
                        double[:, ::1] damper) except -1:

    cdef:
        double [:] ELocal =  np.empty(3, dtype=np.double)
        double [:] SLocal =  np.empty(3, dtype=np.double)

        Py_ssize_t el, index = 0, i, j
        double S1, S2, SE, KuDot

        # wrinkling model variables
        double [:] ETotal = np.empty(3, dtype=np.double)

        double theta

    Fint[...] = 0
    data[...] = 0
    diagK[...] = 0

    if dim == 2:

        # Loop over all cable elements
        for el in range(nelemsCable):

            # if 0 -> bar element, if 1 -> cable element
            wrinklingFlag = NCable[el, 3]

            cable2DFintAndK(X, Y, NCable, L0[el], el, area2[el], E2[el], index,
                            Fint, data, diagK, order, &SE, wrinklingFlag)

            # increase index by length of entries in local stiffness matrix Kloc (6 x 6)
            index += 16

        # Loop over all membrane elements
        for el in range(nelemsMem):
            print("not up to date with wrinkling model")
            # determine strain
            membrane2DStrain(X, Y, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], ELocal)

            # total strain = elastic strain + wrinkling strain
            ETotal[0] = ELocal[0] + Ew[el, 0]
            ETotal[1] = ELocal[1] + Ew[el, 1]
            ETotal[2] = ELocal[2] + Ew[el, 2]

            # determine stressVoigt, S1, S2, theta
            # membraneStress(ELocal, SLocal, &S1, &S2, &theta, E3[el], nu[el], thetaVec[el])

            ## check wrinkling criterion, update SLocal, ELocal, Ew
            if NMem[el, 4] == 1:
                membraneWrinklingJarasjarungkiat(ELocal, SLocal, state, S2, theta, el, E3[el], nu[el],
                                             P, alpha_array, wrinkling_iter, iter_goal, sigma_max)

            # determine KMem
            membrane2DKmat(X, Y, NMem, SLocal, Fint, t[el], area3[el], p[el], el, J11Vec[el], J22Vec[el],
                           J12Vec[el], E3[el], nu[el], data, diagK, index, order)

            # increase index by length of entries in local stiffness matrix Kloc (9 x 9)
            index += 36

    elif dim == 3:

        # Loop over all cable elements
        for el in range(nelemsCable):

            # if 0 -> bar element, if 1 -> cable element
            wrinklingFlag = NCable[el, 3]

            cable3DFintAndK(X, Y, Z, NCable, L0[el], el, area2[el], E2[el],
                            index, Fint, data, diagK, order, &SE, wrinklingFlag)

            # increase index by length of entries in local stiffness matrix Kloc (6 x 6)
            index += 36

        # Loop over all membrane elements
        for el in range(nelemsMem):

            # determine strain
            membrane3DStrain(X, Y, Z, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], ELocal)

            # total strain = elastic strain + wrinkling strain
            ETotal[0] = ELocal[0] + Ew[el, 0]
            ETotal[1] = ELocal[1] + Ew[el, 1]
            ETotal[2] = ELocal[2] + Ew[el, 2]

            # determine stressVoigt, S1, S2, theta
            # membraneStress(ELocal, SLocal, &S1, &S2, &theta, E3[el], nu[el], thetaVec[el])

            # check wrinkling criterion, update SLocal, ELocal, Ew
            if NMem[el, 4] == 1:
                membraneWrinklingJarasjarungkiat(ELocal, SLocal, state, S2, theta, el, E3[el], nu[el],
                                             P, alpha_array, wrinkling_iter, iter_goal, sigma_max)

            # determine KMem
            membrane3DKmat(X, Y, Z, NMem, SLocal, Fint, t[el], area3[el], p[el], el, J11Vec[el], J22Vec[el],
                           J12Vec[el], E3[el], nu[el], data, diagK, index, order)

            # increase index by length of entries in local stiffness matrix Kloc (9 x 9)
            index += 81

    index = 0

    cdef double [:] D = np.zeros(ndof, dtype=np.double)

    if dim == 2:
        for i in range(np.size(damper,0)):
            V_mag = np.sqrt(uDot[2*(int(damper[i,1])+1)-2] ** 2 +
                            uDot[2*(int(damper[i,1])+1)-1] ** 2)
            # Dx, Dy
            D[2*(int(damper[i,1])+1)-2] += damper[i,2] * uDot[2*(int(damper[i,1])+1)-2] * V_mag
            D[2*(int(damper[i,1])+1)-1] += damper[i,2] * uDot[2*(int(damper[i,1])+1)-1] * V_mag

    elif dim == 3:
        for i in range(np.size(damper,0)):
            V_mag = np.sqrt(uDot[3*(int(damper[i,1])+1)-3] ** 2 +
                            uDot[3*(int(damper[i,1])+1)-2] ** 2 +
                            uDot[3*(int(damper[i,1])+1)-1] ** 2)
            # Dx, Dy, Dz
            D[3*(int(damper[i,1])+1)-3] += damper[i,2] * uDot[3*(int(damper[i,1])+1)-3] * V_mag
            D[3*(int(damper[i,1])+1)-2] += damper[i,2] * uDot[3*(int(damper[i,1])+1)-2] * V_mag
            D[3*(int(damper[i,1])+1)-1] += damper[i,2] * uDot[3*(int(damper[i,1])+1)-1] * V_mag

    # compute nodal acceleration
    for i in range(ndof):

        KuDot = 0
        # TODO: this is most probably not the correct. Rather use Fint instead...
        for j in range(indptr[i + 1] - indptr[i]):

            KuDot += data[index] * uDot[indices[index]]
            index += 1

        uDotDot[i] = Minv[i] * (RHS[i] + D[i] - Fint[i]) - alpha * uDot[i] - beta * Minv[i] * KuDot


cdef int update_internal_force_vector(double [:] Fint,
                                      unsigned int dim,
                                      unsigned int nelemsCable,
                                      unsigned int nelemsMem,
                                      int [:, ::1] NCable,
                                      int [:, ::1] NMem,
                                      double [:] X,
                                      double [:] Y,
                                      double [:] Z,
                                      long double [:] J11Vec,
                                      long double [:] J22Vec,
                                      long double [:] J12Vec,
                                      double [:] E2,
                                      double [:] L0,
                                      double [:] area2,
                                      double [:] E3,
                                      double [:] nu,
                                      double [:] area3,
                                      double [:] t,
                                      double [:] theta_vec,
                                      unsigned int [:] state,
                                      double [:] P,
                                      double [:] alpha_array,
                                      double [:, ::1] Ew,
                                      unsigned int wrinkling_iter,
                                      unsigned int iter_goal,
                                      double sigma_max,
                                      str wrinkling_model,
                                      double [:] V,
                                      double beta) except -1:

    cdef:
        double [:] ELocal =  np.empty(3, dtype=np.double)
        double [:] SLocal =  np.empty(3, dtype=np.double)

        # wrinkling model variables
        double [:] ETotal = np.empty(3, dtype=np.double)

        Py_ssize_t el
        double S1, S2
        double theta

        unsigned int wrinklingFlag

    Fint[...] = 0

    if wrinkling_model == "Jarasjarungkiat":

        if dim == 2:

            # Loop over all cable elements
            for el in range(nelemsCable):

                # if 0 -> bar element, if 1 -> cable element
                wrinklingFlag = NCable[el, 3]

                cable2D_internal_force_vector(X, Y, NCable, L0[el], el, area2[el], E2[el],
                                              Fint, wrinklingFlag)

            # Loop over all membrane elements
            for el in range(nelemsMem):

                # determine strain
                membrane2DStrain(X, Y, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], ELocal)

                # determine stressVoigt, S1, S2, theta
                # membraneStress(ELocal, SLocal, &S1, &S2, &theta, E3[el], nu[el], theta_vec[el])

                if NMem[el, 4] == 1:
                    membraneWrinklingJarasjarungkiat(ELocal, SLocal, state, S2, theta, el, E3[el], nu[el],
                                                    P, alpha_array, wrinkling_iter, iter_goal, sigma_max)

                # internal force vector
                membrane2D_internal_force_vector(Fint, J11Vec[el], J22Vec[el], J12Vec[el], NMem,
                                                 el, X, Y, SLocal, area3[el], t[el])

        elif dim == 3:

            # Loop over all cable elements
            for el in range(nelemsCable):

                # if 0 -> bar element, if 1 -> cable element
                wrinklingFlag = NCable[el, 3]

                cable3D_internal_force_vector(X, Y, Z, NCable, L0[el], el, area2[el], E2[el],
                                              Fint, wrinklingFlag)

            # Loop over all membrane elements
            for el in range(nelemsMem):

                # determine strain
                membrane3DStrain(X, Y, Z, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], ELocal)

                # determine stressVoigt, S1, S2, theta
                # membraneStress(ELocal, SLocal, &S1, &S2, &theta, E3[el], nu[el], theta_vec[el])

                if NMem[el, 4] == 1:
                    # correct stress states
                    membraneWrinklingJarasjarungkiat(ELocal, SLocal, state, S2, theta, el, E3[el], nu[el],
                                                     P, alpha_array, wrinkling_iter, iter_goal, sigma_max)

                # internal force vector
                membrane3D_internal_force_vector(Fint, J11Vec[el], J22Vec[el], J12Vec[el], NMem,
                                                 el, X, Y, Z, SLocal, area3[el], t[el],
                                                 E3[el], nu[el], V, beta)

    else:

        if dim == 2:

            # Loop over all cable elements
            for el in range(nelemsCable):

                # if 0 -> bar element, if 1 -> cable element
                wrinklingFlag = NCable[el, 3]



                cable2D_internal_force_vector(X, Y, NCable, L0[el], el, area2[el], E2[el],
                                              Fint, wrinklingFlag)

            # Loop over all membrane elements
            for el in range(nelemsMem):

                # determine strain
                membrane2DStrain(X, Y, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], ELocal)

                # determine stressVoigt, S1, S2, theta
                # membraneStress(ELocal, SLocal, &S1, &S2, &theta, E3[el], nu[el], theta_vec[el])

                # internal force vector
                membrane2D_internal_force_vector(Fint, J11Vec[el], J22Vec[el], J12Vec[el], NMem,
                                                 el, X, Y, SLocal, area3[el], t[el])

        elif dim == 3:

            # Loop over all cable elements
            for el in range(nelemsCable):

                # if 0 -> bar element, if 1 -> cable element
                wrinklingFlag = NCable[el, 3]

                cable3D_internal_force_vector(X, Y, Z, NCable, L0[el], el, area2[el], E2[el],
                                              Fint, wrinklingFlag)

            # Loop over all membrane elements
            for el in range(nelemsMem):

                # determine strain
                membrane3DStrain(X, Y, Z, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el], ELocal)

                # determine stressVoigt, S1, S2, theta
                # membraneStress(ELocal, SLocal, &S1, &S2, &theta, E3[el], nu[el], theta_vec[el])

                # internal force vector
                membrane3D_internal_force_vector(Fint, J11Vec[el], J22Vec[el], J12Vec[el], NMem,
                                                 el, X, Y, Z, SLocal, area3[el], t[el],
                                                 E3[el], nu[el], V, beta)
