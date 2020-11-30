# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
cimport cython

from mem4py.helper.PK2toCauchy cimport PK2toCauchy
from mem4py.elements.membrane cimport membrane2DStrain
from mem4py.elements.membrane cimport membrane3DStrain
from mem4py.elements.membrane cimport membraneStress
from mem4py.elements.membrane cimport membraneWrinklingJarasjarungkiat
from mem4py.elements.cable cimport cable2DCauchyStress
from mem4py.elements.cable cimport cable3DCauchyStress


cdef extern from "math.h":
    double sqrt(double m)
    double atan2(double m, double n)
    double cos(double m)
    double sin(double m)


cdef int computeStress(double [:] X,
                       double [:] Y,
                       double [:] Z,
                       int [:, ::1] NMem,
                       int [:, ::1] NCable,
                       unsigned int nelemsCable,
                       unsigned int nelemsMem,
                       long double [:] J11Vec,
                       long double [:] J12Vec,
                       long double [:] J22Vec,
                       double [:] E_mod_3,
                       double [:] nu,
                       double [:] E_mod_2,
                       double [:] area2,
                       double [:, ::1] S,
                       double [:, ::1] cauchy,
                       double [:] sigma_cable,
                       double [:] S1,
                       double [:] S2,
                       double [:] E1,
                       double [:] E2,
                       double [:, ::1] E,
                       double [:] eps_cable,
                       double [:] thetaVec,
                       double [:] VMS,
                       double [:] area3,
                       double [:] L0,
                       double [:, ::1] Ew,
                       unsigned int [:] state_cable,
                       unsigned int [:] state_mem,
                       unsigned int wrinkling,
                       unsigned int dim,
                       double [:] P,
                       double [:] alpha_array,
                       double sigma_max,
                       str wrinkling_model) except -1:
    """
    
    :param X:           X-coordinates (current configuration)
    :param Y:           Y-coordinates (current configuration)
    :param Z:           Z-coordinates (current configuration)
    :param N:           element connectivity matrix
    :param nelems:      total number of elements
    :param J11Vec:      transformation parameters between initial and current configuration
    :param J12Vec:      transformation parameters between initial and current configuration
    :param J22Vec:      transformation parameters between initial and current configuration
    :param E:           Young's modulus
    :param poisson:     Poisson's ratio
    :param S:           Cauchy stress matrix (sigmaxx, sigmayy, tauxy)
    :param Sp:          principal stress matrix (S1, S2, S_cable)
    :param E:    strain matrix in Voigt form (Exx, Eyy, Exy)
    :param thetaVec:    angle between local and fibre axis
    :param area3:     element areas
    
    :return:            void, stress and strain vectors are filled in memory view
    """

    cdef:

        Py_ssize_t el, i

        double S1_element, S2_element, theta, stressCable, c, s
        double strainCable

        unsigned int [:] allDofMem = np.empty(3 * dim, dtype=np.uintc)

        double [:] SFibre = np.zeros(3, dtype=np.double)
        double [:] SLocal = np.zeros(3, dtype=np.double)
        double [:] ELocal = np.zeros(3, dtype=np.double)
        double [:] EFibre = np.zeros(3, dtype=np.double)
        double [:] ETotal = np.zeros(3, dtype=np.double)
        double [:] cauchyLocal = np.zeros(3, dtype=np.double)

        double [:, :] PK2 = np.empty((2, 2), dtype=np.double)

    if dim == 2:

        for el in range(nelemsCable):

             # if 0 -> bar element, if 1 -> cable element
            wrinkling = NCable[el, 3]

            cable2DCauchyStress(X, Y, NCable, L0[el], E_mod_2[el], &strainCable,
                                &stressCable, el, wrinkling)

            eps_cable[el] = strainCable

            sigma_cable[el] = stressCable

    elif dim == 3:

        for el in range(nelemsCable):

            # if 0 -> bar element, if 1 -> cable element
            wrinkling = NCable[el, 3]

            cable3DCauchyStress(X, Y, Z, NCable, L0[el], E_mod_2[el], &strainCable,
                                &stressCable, el, wrinkling)

            eps_cable[el] = strainCable

            sigma_cable[el] = stressCable

            if wrinkling == 1 and stressCable <= 0:

                state_cable[el] = 0

            else:

                state_cable[el] = 2


    for el in range(nelemsMem):

        if dim == 2:

            # compute elastic strain and element dofs
            membrane2DStrain(X, Y, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el],
                             ELocal)
        elif dim == 3:
            # compute elastic strain and element dofs
            membrane3DStrain(X, Y, Z, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el],
                             ELocal)

        c = cos(-thetaVec[el])
        s = sin(-thetaVec[el])

        # determine stressVoigt, S1, S2, theta
        membraneStress(ELocal, SLocal, &S1_element, &S2_element, &theta, E_mod_3[el], nu[el])

        if NMem[el, 4] == 1:
            # check wrinkling criterion, update SLocal, ELocal, Ew
            membraneWrinklingJarasjarungkiat(ELocal, SLocal, state_mem, S2_element, theta, el,  E_mod_3[el],
                                             nu[el], P, alpha_array, 1, 2, sigma_max)

        # fibre strain
        EFibre[0] = c * c * ELocal[0] + s * s * ELocal[1] + s * c * ELocal[2]
        EFibre[1] = s * s * ELocal[0] + c * c * ELocal[1] - s * c * ELocal[2]
        EFibre[2] = - 2 * s * c * ELocal[0] + 2 * s * c * ELocal[1] + (c * c - s * s) * ELocal[2]

        # determine local stress
        SFibre[0] =  c * c * SLocal[0] + s * s * SLocal[1] + 2 * s * c * SLocal[2]
        SFibre[1] = s * s * SLocal[0] + c * c * SLocal[1] - 2 * s * c * SLocal[2]
        SFibre[2] =  - s * c * SLocal[0] + s * c * SLocal[1] + (c * c - s * s) * SLocal[2]

        # PK2 tensor (symmetric tensor)
        PK2[0, 0] = SFibre[0]
        PK2[1, 1] = SFibre[1]
        PK2[1, 0] = SFibre[2]
        PK2[0, 1] = SFibre[2]

        # convert PK2 to cauchy stress
        PK2toCauchy(X, Y, Z, J11Vec[el], J22Vec[el], J12Vec[el], NMem, el, area3[el], PK2, cauchyLocal)

        for i in range(3):
            S[nelemsCable + el, i] = SFibre[i]
            cauchy[nelemsCable + el, i] = cauchyLocal[i]
            E[nelemsCable + el, i] = EFibre[i]

        # Principal strain (epsilon_1, epsilon_2, phi)
        a = (EFibre[ 0] + EFibre[1]) / 2
        b = EFibre[2] * EFibre[2] / 4 - \
            EFibre[0] * EFibre[1]

        # epsilon1
        E1[nelemsCable + el] = a + sqrt(a * a + b)

        # epsilon2
        E2[nelemsCable + el] = a - sqrt(a * a + b)

        # phi
        # Ep[nelemsCable + el, 2] = (0.5 * atan2(2 * EFibre[2] ,
        #                         (EFibre[0] - EFibre[1]))) * 180 / np.pi

        # Principal stress (sigma_1, sigma_2, phi)
        a = (cauchy[nelemsCable + el, 0] + cauchy[nelemsCable + el, 1]) / 2
        b = cauchy[nelemsCable + el, 2] * cauchy[nelemsCable + el, 2] - \
            cauchy[nelemsCable + el, 0] * cauchy[nelemsCable + el, 1]

        # sigma1
        S1[nelemsCable + el] = a + sqrt(a * a + b)

        # sigma2
        S2[nelemsCable + el] = a - sqrt(a * a + b)

        # phi
        # Sp[nelemsCable + el, 2] = (0.5 * atan2(2 * SFibre[2] ,
        #                         (SFibre[0] - SFibre[1]))) * 180 / np.pi

        # Von Mises stress
        VMS[nelemsCable + el] = sqrt((cauchy[nelemsCable + el, 0] + cauchy[nelemsCable + el, 1]) *
                                   (cauchy[nelemsCable + el, 0] + cauchy[nelemsCable + el, 1]) -
                                   3 * (cauchy[nelemsCable + el, 0] * cauchy[nelemsCable + el, 1] -
                                        cauchy[nelemsCable + el, 2] * cauchy[nelemsCable + el, 2]))