import numpy as np
cimport numpy as np
cimport cython

# C-import all helper functions
from src.readMsh cimport readMsh
from src.helper.checkInput cimport checkInput
from src.solver.DRIso cimport DRIsotropic3D
from src.solver.DRIso cimport DRIsotropic2D
from src.solver.dynamicExplicit cimport solve3DDynamicExplicit
from src.solver.static2D cimport static2D
from src.helper.cellCentre cimport cellCentre
from src.writeOutput cimport writeVTK2D
from src.writeOutput cimport writeVTK3D

# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
class Mem4py:

    """
    class mem4py with methods:

    - readMesh: Read msh file and save nodal coordinates and element connectivity

    - dirichletHandler: Helper functions to handle Dirichlet boundary conditions

    - assembler: Assembler functions to generate system of equations

    - solver: Linear and nonlinear solvers

    - postprocessor: Determine element stress and save in .vtk file

    """
    # TODO:

    def __init__(self, problem):

        props = {}
        load = {}
        solverOptions = {}

        # check input dict "problem"
        checkInput(problem)

        # problem dimension
        self.dim = problem["dim"]

        # msh file name
        self.inputName = problem["msh"]

        # scaling factor for mesh input file (default = 1)
        if not "scale" in problem:
            self.scale = 1.
        else:
            self.scale = problem["scale"]

        # Initialize solver variables and other constants
        self.nnodes = 0
        self.ndof = 0
        self.nelems = 0
        self.nelemsMem = 0
        self.nelemsBar = 0
        self.nelemsBeam = 0
        self.strainEnergy = 0.
        self.KE = 0.

        self.X0 = []
        self.Y0 = []
        self.Z0 = []
        self.X = []
        self.Y = []
        self.Z = []
        self.Nm = []
        self.Nb = []
        self.u = []
        self.V = []
        self.dofFixed = []
        self.prescribedDof = []
        self.prescribedDisplacement = []
        self.loadedBCNodes = None
        self.loadedBCEdges = None
        self.elPressurised = None
        self.nPressurised = []
        self.areaVec = []
        self.LBar = []
        self.J11Vec = []
        self.J12Vec = []
        self.J22Vec = []
        self.thetaVec = []
        self.matDir = np.array([1, 0, 0])
        self.matDirType = "global"
        self.matDirAngle = None
        self.p = []
        self.dt = 0
        self.time = 0
        self.tLength = 0

        self.Ew = []
        self.RF = []

        # distributed load from OpenFOAM
        self.Sx = []
        self.Sy = []
        self.Sz = []

        # cell centres in matrix format
        self.cc = []

        # autoWrite, if true write vtk after converged in solve()
        if not "autoWrite" in problem:
            self.autoWrite = True
        elif problem["autoWrite"] is False:
            self.autoWrite = False

        # Material properties
        props["mat"] = problem["mat"]

        # Membrane Young's modulus
        if not "EMem" in problem:
            props["EMem"] = 0.
        else:
            props["EMem"] = problem["EMem"]

        # Membrane Poisson's ratio
        if not "nu" in problem:
            props["nu"] = 0.
        else:
            props["nu"] = problem["nu"]

        # Membrane thickness
        if not "t" in problem:
            props["t"] = 0.
        else:
            props["t"] = problem["t"]

        # Membrane density
        if not "rhoMem" in problem:
            props["rhoMem"] = 0.
        else:
            props["rhoMem"] = problem["rhoMem"]

        # Bar Young's modulus
        if not "EBar" in problem:
            props["EBar"] = 0.
        else:
            props["EBar"] = problem["EBar"]

        # Bar cross-sectional area
        if not "areaBar" in problem:
            props["areaBar"] = 0.
        else:
            props["areaBar"] = problem["areaBar"]

        # Bar density
        if not "rhoBar" in problem:
            props["rhoBar"] = 0.
        else:
            props["rhoBar"] = problem["rhoBar"]

        # Rayleigh damping parameter alpha
        if not "alpha" in problem:
            props["alpha"] = 0.
        else:
            props["alpha"] = problem["alpha"]

        # Rayleigh damping parameter alpha
        if not "beta" in problem:
            props["beta"] = 0.
        else:
            props["beta"] = problem["beta"]

        # time step dt
        if not "dt" in problem:
            props["dt"] = 1E-4
        else:
            props["dt"] = problem["dt"]

        # simulation time T
        if not "T" in problem:
            props["T"] = 1
        else:
            props["T"] = problem["T"]

        # Nodal load fX
        if not "fX" in problem:
            load["fX"] = 0.
        else:
            load["fX"] = problem["fX"]

        # Nodal load fY
        if not "fY" in problem:
            load["fY"] = 0.
        else:
            load["fY"] = problem["fY"]

        # Nodal load fZ
        if not "fZ" in problem:
            load["fZ"] = 0.
        else:
            load["fZ"] = problem["fZ"]

        # Gravity or other acceleration field
        if not "gravity" in problem:
            self.gravity = False
            load["gX"] = 0.
            load["gY"] = 0.
            load["gZ"] = 0.
        else:
            if problem["gravity"] is False:
                self.gravity = False
                load["gX"] = 0.
                load["gY"] = 0.
                load["gZ"] = 0.
            if problem["gravity"] is True and problem["gX"]:
                self.gravity = True
                load["gX"] = problem["gX"]
            if problem["gravity"] is True and problem["gY"]:
                self.gravity = True
                load["gY"] = problem["gY"]
            if problem["gravity"] is True and problem["gZ"]:
                self.gravity = True
                load["gZ"] = problem["gZ"]

        # Normal edge load
        if not "edgeNormal" in problem:
            load["edgeNormal"] = 0.
        else:
            load["edgeNormal"] = problem["edgeNormal"]

        # Edge load in x-direction
        if not "edgeX" in problem:
            load["edgeX"] = 0.
        else:
            load["edgeX"] = problem["edgeX"]

        # Edge load in y-direction
        if not "edgeY" in problem:
            load["edgeY"] = 0.
        else:
            load["edgeY"] = problem["edgeY"]

        # Edge load in z-direction
        if not "edgeZ" in problem:
            load["edgeZ"] = 0.
        else:
            load["edgeZ"] = problem["edgeZ"]

        # Pressure load acting on element surface
        if not "p" in problem:
            load["p"] = 0.
        else:
            load["p"] = problem["p"]

        # Nonlinear analysis (Only for static solver)
        if not "NL" in problem:
            solverOptions["NL"] = False
        else:
            solverOptions["NL"] = problem["NL"]

        # print convergence information
        if not "printConvergence" in problem:
            solverOptions["printConvergence"] = True
        else:
            solverOptions["printConvergence"] = problem["printConvergence"]

        # Newton-Raphson convergence criterion
        if not "epsilonNR" in problem:
            if solverOptions["NL"] is True:
                print("epsilonNR not defined. Using default: 1E-10")
            solverOptions["epsilonNR"] = 1E-10
        else:
            solverOptions["epsilonNR"] = problem["epsilonNR"]

        # Newton-Raphson maximum number of iterations
        if not "maxIter" in problem:
            if solverOptions["NL"] is True:
                print("maxIter not defined. Using default: 100")
            solverOptions["maxIter"] = 100
        else:
            solverOptions["maxIter"] = problem["maxIter"]

        # Number of load step increments (linear)
        if not "nLoadSteps" in problem:
            solverOptions["nLoadSteps"] = 1
        else:
            solverOptions["nLoadSteps"] = problem["nLoadSteps"]

        # Static solver type (linear or nonlinear)
        if not "static" in problem:
            solverOptions["static"] = False
        else:
            solverOptions["static"] = problem["static"]

        # Follower load
        if not "follower" in problem:
            solverOptions["follower"] = False
        else:
            solverOptions["follower"] = problem["follower"]

        # Dynamic explicit solver type
        if not "dynExplicit" in problem:
            solverOptions["dynExplicit"] = False
        else:
            solverOptions["dynExplicit"] = problem["dynExplicit"]

        # Kinetic dynamic relaxation solver type
        if not "DR" in problem:
            solverOptions["DR"] = False
        else:
            solverOptions["DR"] = problem["DR"]

        # Wrinkling model
        if not "wrinkling" in problem:
            solverOptions["wrinkling"] = False
        else:
            solverOptions["wrinkling"] = problem["wrinkling"]

        # Kinetic dynamic relaxation force convergence criterion
        if not "epsilonR" in problem:
            if solverOptions["DR"] is True:
                print("epsilonR not defined. Using default: 1E-5")
            solverOptions["epsilonR"] = 1E-5
        else:
            solverOptions["epsilonR"] = problem["epsilonR"]

        # Kinetic dynamic relaxation kinetic energy convergence criterion
        if not "epsilonKE" in problem:
            if solverOptions["DR"] is True:
                print("epsilonKE not defined. Using default: 1E-10")
            solverOptions["epsilonKE"] = 1E-10
        else:
            solverOptions["epsilonKE"] = problem["epsilonKE"]


        # solverOptions["WRITE"] = 0.001
        # solverOptions["TIMESTEP"] = 0.00001
        # solverOptions["SIMULATIONTIME"] = 2.0

        # solverOptions["INTEGRATOR"] = 'lsoda'
        # solverOptions["INTEGRATORMETHOD"] = ''

        # Insert dicts into object attributes
        self.props = props
        self.load = load
        self.solverOptions = solverOptions

        # read msh
        self.readMesh()

    def readMesh(self):

        readMsh(self)

        if self.nelems == 0:
            raise Exception("No elements could be read/found. Please check input file")
        elif self.nelemsBar == 0:
            print("Finished reading \"{}\"\n"
                  "{} membrane elements with {} degrees of freedom in R{} found".format(self.inputName ,
                                                                                        self.nelemsMem,
                                                                                        self.nnodes * self.dim,
                                                                                        self.dim))
        elif self.nelemsMem == 0:
            print("Finished reading \"{}\"\n"
                  "{} bar elements with {} degrees of freedom in R{} found".format(self.inputName ,
                                                                                   self.nelemsBar,
                                                                                   self.nnodes * self.dim,
                                                                                   self.dim))
        else:
            print("Finished reading \"{}\"\n"
                  "{} bar elements and {} membrane elements with {} degrees of freedom in R{} found".format(self.inputName,
                                                                                                            self.nelemsBar,
                                                                                                            self.nelemsMem,
                                                                                                            self.nnodes * self.dim,
                                                                                                            self.dim))


    def solve(self):

        if self.dim == 2 and self.solverOptions["DR"] is False and self.solverOptions["NL"] is False:
            # Solve 2D static linear
            print("Solving linear static")
            static2D(self)
        elif self.dim == 2 and self.solverOptions["DR"] is False and self.solverOptions["NL"] is True:
            # Solve 2D static nonlinear (Newton-Raphson)
            print("Running nonlinear static")
            static2D(self)
        elif self.dim == 2 and self.solverOptions["DR"] is True:
            # Solve 2D dynamic relaxation
            print("Running dynamic relaxation")
            DRIsotropic2D(self)
        elif self.dim == 3 and self.solverOptions["DR"] is True:
            print("Starting DR Alamatian with isotropic elastic material")
            DRIsotropic3D(self)

        elif self.dim == 3 and self.solverOptions["dynExplicit"] is True:
            print("Starting explicit dynamic solver with wrinkling model")
            solve3DDynamicExplicit(self)
        else:
            raise Exception("No solver type defined.")

        print("***** ALL DONE *****")


    # cell centres of current configuration
    def computeCellCentres(self):

        cdef unsigned int [:] dofList = self.elPressurised
        cdef double [:] X = self.X
        cdef double [:] Y = self.Y
        cdef double [:] Z = self.Z
        cdef int [:, ::1] Nm = self.Nm
        cdef double [:, ::1] cc = self.cc

        cellCentre(dofList, X, Y, Z, Nm, cc)


    # method callable in python to write current output
    def writeVTK(self):

        if self.dim == 2:
            writeVTK2D(self)
        elif self.dim == 3:
            writeVTK3D(self)
        else:
            raise Exception("Cannot write output, no dimension defined")
