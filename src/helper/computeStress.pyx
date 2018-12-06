import numpy as np
cimport numpy as np
cimport cython

from src.materialModels.StVenantIso cimport CmatStVenantIsotropic

from src.helper.PK2toCauchy cimport PK2toCauchy

from src.elements.membrane3D cimport membrane3DStrain
from src.elements.membrane3D cimport membrane3DStress

from src.elements.membrane2D cimport membrane2DStrain
from src.elements.membrane2D cimport membrane2DStress

from src.elements.bar3D cimport bar3DCauchyStress

from src.elements.bar2D cimport bar2DCauchyStress

from src.ceygen.math cimport dot_mv


cdef extern from "math.h":
    double sqrt(double m)
    double atan2(double m, double n)
    double cos(double m)
    double sin(double m)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int computeStress2D(double [:] X,
                          double [:] Y,
                          int [:, ::1] NMem,
                          int [:, ::1] NBar,
                          unsigned int nelemsBar,
                          unsigned int nelemsMem,
                          double [:] J11Vec,
                          double [:] J12Vec,
                          double [:] J22Vec,
                          double EMem,
                          double poisson,
                          double EBar,
                          double areaBar,
                          double [:, ::1] S,
                          double [:, ::1] Sp,
                          double [:, ::1] Ep,
                          double [:, ::1] Eelastic,
                          double [:] thetaVec,
                          double [:] VMS,
                          double [:] areaVec,
                          double [:] L0,
                          double [:, ::1] Ew,
                          unsigned int [:] state) except -1:
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
    :param Sp:          principal stress matrix (S1, S2, phi)
    :param Eelastic:    strain matrix in Voigt form (Exx, Eyy, Exy)
    :param thetaVec:    angle between local and fibre axis
    :param areaVec:     element areas
    
    :return:            void, stress and strain vectors are filled in memory view
    """

    cdef:

        unsigned int el
        double S1, S2, theta, strainBar, stressBar

        unsigned int [:] allDofMem = np.empty(6, dtype=np.uintc)
        double [:] SFibre = np.zeros(3, dtype=np.double)
        double [:] SLocal = np.zeros(3, dtype=np.double)
        double [:] ELocal = np.zeros(3, dtype=np.double)
        double [:] EFibre = np.zeros(3, dtype=np.double)
        double [:] ETotal = np.zeros(3, dtype=np.double)
        double [:] cauchyLocal = np.zeros(3, dtype=np.double)

        double [:, :] Cmat = np.zeros((3, 3), dtype=np.double)
        double [:, :] PK2 = np.empty((2, 2), dtype=np.double)
        double [:, :] T = np.empty((3, 3), dtype=np.double)

        double [:] Z = np.zeros(len(X), dtype=np.double)


    # constitutive matrix
    CmatStVenantIsotropic(EMem, poisson, Cmat)

    for el in range(nelemsBar):

        bar2DCauchyStress(X, Y, NBar, L0[el], EBar, &strainBar, &stressBar, el)

        Eelastic[el, 0] = strainBar
        Ep[el, 0] = strainBar

        S[el, 0] = stressBar
        Sp[el, 0] = stressBar

    for el in range(nelemsMem):

        # compute elastic strain and element dofs
        membrane2DStrain(X, Y, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el],
                         allDofMem, ELocal)

        # rotate strain tensor to material coordinates
        T[0, 0] = cos(thetaVec[el]) * cos(thetaVec[el])
        T[0, 1] = sin(thetaVec[el]) * sin(thetaVec[el])
        T[0, 2] = sin(thetaVec[el]) * cos(thetaVec[el])

        T[1, 0] = sin(thetaVec[el]) * sin(thetaVec[el])
        T[1, 1] = cos(thetaVec[el]) * cos(thetaVec[el])
        T[1, 2] = - sin(thetaVec[el]) * cos(thetaVec[el])

        T[2, 0] = - 2 * sin(thetaVec[el]) * cos(thetaVec[el])
        T[2, 1] = 2 * sin(thetaVec[el]) * cos(thetaVec[el])
        T[2, 2] = cos(thetaVec[el]) * cos(thetaVec[el]) - sin(thetaVec[el]) * sin(thetaVec[el])

        # total strain = elastic strain + wrinkling strain
        ETotal[0] = ELocal[0] + Ew[nelemsBar + el, 0]
        ETotal[1] = ELocal[1] + Ew[nelemsBar + el, 1]
        ETotal[2] = ELocal[2] + Ew[nelemsBar + el, 2]

        # determine stressVoigt, S1, S2, theta
        membrane2DStress(Cmat, ELocal, SLocal, &S1, &S2, &theta)

        # fibre strain
        dot_mv(T, ELocal, EFibre)

        # determine fibre stress
        dot_mv(Cmat, EFibre, SFibre)

        # PK2 tensor (symmetric tensor)
        PK2[0, 0] = SFibre[0]
        PK2[1, 1] = SFibre[1]
        PK2[1, 0] = SFibre[2]
        PK2[0, 1] = SFibre[2]

        # convert PK2 to cauchy stress
        PK2toCauchy(X, Y, Z, NMem, el, J11Vec[el], J12Vec[el], J22Vec[el], PK2, cauchyLocal)

        for i in range(3):
            S[nelemsBar + el, i] = cauchyLocal[i]
            # S[nelemsBar + el, i] = SFibre[i]
            Eelastic[nelemsBar + el, i] = EFibre[i]

        # Principal strain (epsilon_1, epsilon_2, phi)
        a = (EFibre[ 0] + EFibre[1]) / 2
        b = EFibre[2] * EFibre[2] - \
            EFibre[0] * EFibre[1]

        # sigma1
        Ep[nelemsBar + el, 0] = a + sqrt(a * a + b)

        # sigma2
        Ep[nelemsBar + el, 1] = a - sqrt(a * a + b)

        # phi
        Ep[nelemsBar + el, 2] = (0.5 * atan2(2 * EFibre[2] ,
                                (EFibre[0] - EFibre[1]))) * 180 / np.pi

        # Principal stress (sigma_1, sigma_2, phi)
        a = (cauchyLocal[0] + cauchyLocal[1]) / 2
        b = cauchyLocal[2] * cauchyLocal[2] - cauchyLocal[0] * cauchyLocal[1]
        # a = (SFibre[0] + SFibre[1]) / 2
        # b = SFibre[2] * SFibre[2] - SFibre[0] * SFibre[1]

        # sigma1
        Sp[nelemsBar + el, 0] = a + sqrt(a * a + b)

        # sigma2
        Sp[nelemsBar + el, 1] = a - sqrt(a * a + b)

        # phi
        # Sp[el, 2] = (0.5 * atan2(2 * cauchyLocal[2] ,
        #                         (cauchyLocal[0] - cauchyLocal[1]))) * 180 / np.pi
        # Sp[nelemsBar + el, 2] = (0.5 * atan2(2 * SFibre[2] ,
        #                         (SFibre[0] - SFibre[1]))) * 180 / np.pi

        # Von Mises stress
        VMS[nelemsBar + el] = sqrt((S[nelemsBar + el, 0] + S[nelemsBar + el, 1]) *
                                   (S[nelemsBar + el, 0] + S[nelemsBar + el, 1]) -
                                   3 * (S[nelemsBar + el, 0] * S[nelemsBar + el, 1] -
                                        S[nelemsBar + el, 2] * S[nelemsBar + el, 2]))


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int computeStress3D(double [:] X,
                          double [:] Y,
                          double [:] Z,
                          int [:, ::1] NMem,
                          int [:, ::1] NBar,
                          unsigned int nelemsBar,
                          unsigned int nelemsMem,
                          double [:] J11Vec,
                          double [:] J12Vec,
                          double [:] J22Vec,
                          double EMem,
                          double poisson,
                          double EBar,
                          double areaBar,
                          double [:, ::1] S,
                          double [:, ::1] Sp,
                          double [:, ::1] Ep,
                          double [:, ::1] Eelastic,
                          double [:] thetaVec,
                          double [:] VMS,
                          double [:] areaVec,
                          double [:] L0,
                          double [:, ::1] Ew,
                          unsigned int [:] state) except -1:
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
    :param Sp:          principal stress matrix (S1, S2, phi)
    :param Eelastic:    strain matrix in Voigt form (Exx, Eyy, Exy)
    :param thetaVec:    angle between local and fibre axis
    :param areaVec:     element areas
    
    :return:            void, stress and strain vectors are filled in memory view
    """

    cdef:

        unsigned int el
        double S1, S2, theta, strainBar, stressBar

        unsigned int [:] allDofMem = np.empty(9, dtype=np.uintc)
        double [:] SFibre = np.zeros(3, dtype=np.double)
        double [:] SLocal = np.zeros(3, dtype=np.double)
        double [:] ELocal = np.zeros(3, dtype=np.double)
        double [:] EFibre = np.zeros(3, dtype=np.double)
        double [:] ETotal = np.zeros(3, dtype=np.double)
        double [:] cauchyLocal = np.zeros(3, dtype=np.double)

        double [:, :] Cmat = np.zeros((3, 3), dtype=np.double)
        double [:, :] PK2 = np.empty((2, 2), dtype=np.double)
        double [:, :] T = np.empty((3, 3), dtype=np.double)

    for el in range(nelemsBar):

        bar3DCauchyStress(X, Y, Z, NBar, L0[el], EBar, &strainBar, &stressBar, el)

    for el in range(nelemsMem):

        # compute elastic strain and element dofs
        membrane3DStrain(X, Y, Z, NMem, el, J11Vec[el], J22Vec[el], J12Vec[el],
                         allDofMem, ELocal)

        # rotate strain tensor to material coordinates
        T[0, 0] = cos(thetaVec[el]) * cos(thetaVec[el])
        T[0, 1] = sin(thetaVec[el]) * sin(thetaVec[el])
        T[0, 2] = sin(thetaVec[el]) * cos(thetaVec[el])

        T[1, 0] = sin(thetaVec[el]) * sin(thetaVec[el])
        T[1, 1] = cos(thetaVec[el]) * cos(thetaVec[el])
        T[1, 2] = - sin(thetaVec[el]) * cos(thetaVec[el])

        T[2, 0] = - 2 * sin(thetaVec[el]) * cos(thetaVec[el])
        T[2, 1] = 2 * sin(thetaVec[el]) * cos(thetaVec[el])
        T[2, 2] = cos(thetaVec[el]) * cos(thetaVec[el]) - sin(thetaVec[el]) * sin(thetaVec[el])

        # constitutive matrix
        CmatStVenantIsotropic(EMem, poisson, Cmat)

        # total strain = elastic strain + wrinkling strain
        ETotal[0] = ELocal[0] + Ew[nelemsBar + el, 0]
        ETotal[1] = ELocal[1] + Ew[nelemsBar + el, 1]
        ETotal[2] = ELocal[2] + Ew[nelemsBar + el, 2]

        # determine stressVoigt, S1, S2, theta
        membrane3DStress(Cmat, ELocal, SLocal, &S1, &S2, &theta)

        dot_mv(T, ELocal, EFibre)

        # determine local stress
        dot_mv(Cmat, EFibre, SFibre)

        # PK2 tensor (symmetric tensor)
        PK2[0, 0] = SFibre[0]
        PK2[1, 1] = SFibre[1]
        PK2[1, 0] = SFibre[2]
        PK2[0, 1] = SFibre[2]

        # convert PK2 to cauchy stress
        PK2toCauchy(X, Y, Z, NMem, el, J11Vec[el], J12Vec[el], J22Vec[el], PK2, cauchyLocal)

        for i in range(3):
            S[nelemsBar + el, i] = cauchyLocal[i]
            # S[el, i] = SFibre[i]
            Eelastic[nelemsBar + el, i] = EFibre[i]

        # Principal strain (epsilon_1, epsilon_2, phi)
        a = (EFibre[ 0] + EFibre[1]) / 2
        b = EFibre[2] * EFibre[2] - \
            EFibre[0] * EFibre[1]

        # sigma1
        Ep[nelemsBar + el, 0] = a + sqrt(a * a + b)

        # sigma2
        Ep[nelemsBar + el, 1] = a - sqrt(a * a + b)

        # phi
        Ep[nelemsBar + el, 2] = (0.5 * atan2(2 * EFibre[2] ,
                                (EFibre[0] - EFibre[1]))) * 180 / np.pi

        # Principal stress (sigma_1, sigma_2, phi)
        a = (cauchyLocal[0] + cauchyLocal[1]) / 2
        b = cauchyLocal[2] * cauchyLocal[2] - cauchyLocal[0] * cauchyLocal[1]
        # a = (SFibre[0] + SFibre[1]) / 2
        # b = SFibre[2] * SFibre[2] - SFibre[0] * SFibre[1]

        # sigma1
        Sp[nelemsBar + el, 0] = a + sqrt(a * a + b)

        # sigma2
        Sp[nelemsBar + el, 1] = a - sqrt(a * a + b)

        # phi
        # Sp[el, 2] = (0.5 * atan2(2 * cauchyLocal[2] ,
        #                         (cauchyLocal[0] - cauchyLocal[1]))) * 180 / np.pi
        Sp[nelemsBar + el, 2] = (0.5 * atan2(2 * SFibre[2] ,
                                (SFibre[0] - SFibre[1]))) * 180 / np.pi

        # Von Mises stress
        VMS[nelemsBar + el] = sqrt((S[nelemsBar + el, 0] + S[nelemsBar + el, 1]) *
                                   (S[nelemsBar + el, 0] + S[nelemsBar + el, 1]) -
                                   3 * (S[nelemsBar + el, 0] * S[nelemsBar + el, 1] -
                                        S[nelemsBar + el, 2] * S[nelemsBar + el, 2]))
