# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
import numpy as np
cimport numpy as np
cimport cython
import os

from src.readMsh cimport readMsh
from src.solver.DR cimport solveKDR
from src.solver.transient cimport solveTransient
from src.helper.cellCentre cimport cellCentre2D, cellCentre3D
from src.writeOutput cimport writeVTK


class Mem4py:

    """
    class mem4py with methods:

    - readMesh: Read msh file and save nodal coordinates and element connectivity

    - dirichletHandler: Helper functions to handle Dirichlet boundary conditions

    - assembler: Assembler functions to generate system of equations

    - solver: Linear and nonlinear solvers

    - postprocessor: Determine element stress and save in .vtk file

    """

    def __init__(self, problem):

        props = {}
        load = {}
        solverOptions = {}

        # check input dict "problem" in main.py
        # checkInput(problem)

        # object with all element sets and their properties
        self.elID = []

        self.elStruc = problem["elStruc"]
        for key in self.elStruc:
            if self.elStruc[key]["set"] == "ELEMENT":
                self.elID.append(0)
            elif self.elStruc[key]["set"] == "BC":
                self.elID.append(1)
            elif self.elStruc[key]["set"] == "LOAD":
                self.elID.append(2)
            elif self.elStruc[key]["set"] == "SET":
                self.elID.append(3)
        # for key in problem["elStruc"]:
        #     self.elStruc[key] = elementStructure(problem["elStruc"].get(key))
        #
        # print(list(self.elStruc))

        # problem dimension
        self.dim = problem["dim"]

        # msh file name
        self.inputName = problem["msh"]

        # vtk result file for restart
        self.restart = False
        if "restart" in problem:
            if problem["restart"] is True:
                self.restart = True
                self.restartName = problem["vtk_name"]

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
        self.nelemsCable = 0
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
        self.Nc = []
        self.u = []
        self.V = []
        self.acc = []
        self.dofFixed = []
        self.prescribedDof = []
        self.prescribedDisplacement = []
        self.loadedBCNodes = None
        self.loadedBCEdges = None
        self.elPressurised = None
        self.nPressurised = []
        self.area3 = []
        self.LCable = []
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
        self.tLength = 5
        self.writeInterval = 1

        self.kinetic_energy = []
        self.strain_energy = []
        self.time_array = []

        self.t = []
        self.area2 = []
        self.rho3 = []
        self.rho2 = []
        self.E3 = []
        self.E2 = []
        self.nu = []

        self.Ew = []
        self.Sw = []
        self.RF = []

        self.RHS = []
        self.Fint = []
        self.M = []

        self.Eelastic = []
        self.Ep = []
        self.S = []
        self.Sp = []
        self.VMS = []
        self.state = []

        # distributed load from OpenFOAM
        self.Sx = []
        self.Sy = []
        self.Sz = []

        # pressure load from external solver
        self.pFSI = []

        # interface element indices for FSI analysis
        self.indFSI = []
        self.indNodesInFSI = []

        # cell centres in matrix format
        self.cc = []

        # contact condition
        self.contactCondition = False
        self.contact = {}
        self.RFContact = []

        # matrix with neighbouring element IDs
        self.neighboursFromElement = []
        self.neighboursFromLine = []

        # TODO: clean up in future
        self.loadedBCNodes_damper = []

        # setNames
        self.setNames = []

        self.output_dir = []

        # constant force vector directly added to RHS, len(ndof)
        self.force_vector = []

        self.KEVec = []
        self.KEpeak = []

        # autoWrite, if true write vtk after converged in solve()
        if not "autoWrite" in problem:
            self.autoWrite = True
        elif problem["autoWrite"] is False:
            self.autoWrite = False

        # write_each_peak, if true write vtk when energy peak is detected in solve() KDR
        if not "write_each_peak" in problem:
            self.write_each_peak = False
        elif problem["write_each_peak"] is True:
            self.write_each_peak = True
        else:
            self.write_each_peak = False

        # write_each_timestep, if true write vtk each time step in solve()
        if not "write_each_timestep" in problem:
            self.write_each_timestep = False
        elif problem["write_each_timestep"] is True:
            self.write_each_timestep = True
        elif problem["write_each_timestep"] is False:
            self.write_each_timestep = False
        else:
            raise Exception("Define write_each_timestep with True/False only!")

        if not "silent" in problem:
            self.silent = False
        else:
            self.silent = problem["silent"]

        if "writeInterval" in problem:
            self.writeInterval = problem["writeInterval"]

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

        # Gravity or other acceleration field
        if not "gravity" in problem:
            self.gravity = False
            load["g"] = [0, 0, 0]
        else:
            if problem["gravity"] is False:
                self.gravity = False
                load["g"] = [0, 0, 0]
            if problem["gravity"] is True and "g" in problem:
                self.gravity = True
                load["g"] = problem["g"]

        # Nonlinear analysis (Only for static solver)
        if not "NL" in problem:
            solverOptions["NL"] = False
        else:
            solverOptions["NL"] = problem["NL"]

        if not "line_search" in problem:
            solverOptions["line_search"] = False
        else:
            solverOptions["line_search"] = problem["line_search"]

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

        # Wrinkling strain energy convergence criterion
        if not "epsilonMU" in problem:
            # print("epsilonMU not defined. Using default: 1E-6")
            solverOptions["epsilonMU"] = 1E-6
        else:
            solverOptions["epsilonMU"] = problem["epsilonMU"]

        # Change in strain energy convergence criterion
        if not "epsilonSE" in problem:
            # print("epsilonSE not defined. Using default: 1E-4")
            solverOptions["epsilonSE"] = 1E-4
        else:
            solverOptions["epsilonSE"] = problem["epsilonSE"]

        if not "max_iter" in problem:
            solverOptions["max_iter"] = 100
        else:
            solverOptions["max_iter"] = problem["max_iter"]

        # Newton-Raphson maximum number of iterations
        if not "maxIter" in problem:
            if solverOptions["NL"] is True:
                print("maxIter not defined. Using default: 100")
            solverOptions["maxIter"] = 10
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

        if not "beta_visc" in problem:
            solverOptions["beta_visc"] = 0.
        else:
            solverOptions["beta_visc"] = problem["beta_visc"]

        if not "method" in problem:
            solverOptions["method"] = "none"
        else:
            solverOptions["method"] = problem["method"]

        if not "lam" in problem:
            solverOptions["lam"] = 1.
        else:
            solverOptions["lam"] = problem["lam"]

        if not "solver" in problem:
            raise Exception("Not solver defined")

        self.solver = problem["solver"]

        solverOptions["max_iter_NR_implicit"] = 100
        if "max_iter_NR_implicit" in problem:
            solverOptions["max_iter_NR_implicit"] = problem["max_iter_NR_implicit"]

        solverOptions["max_iter_dt_implicit"] = 1000
        if "max_iter_dt_implicit" in problem:
            solverOptions["max_iter_dt_implicit"] = problem["max_iter_dt_implicit"]

        solverOptions["epsilon_NR_implicit"] = 1E-10
        if "epsilon_NR_implicit" in problem:
            solverOptions["epsilon_NR_implicit"] = problem["epsilon_NR_implicit"]

        solverOptions["alpha_m_implicit"] = 1.
        if "alpha_m_implicit" in problem:
            solverOptions["alpha_m_implicit"] = problem["alpha_m_implicit"]

        solverOptions["alpha_f_implicit"] = 1.
        if "alpha_f_implicit" in problem:
            solverOptions["alpha_f_implicit"] = problem["alpha_f_implicit"]

        solverOptions["beta_implicit"] = 0.
        if "beta_implicit" in problem:
            solverOptions["beta_implicit"] = problem["beta_implicit"]

        solverOptions["gamma_implicit"] = 0.
        if "gamma_implicit" in problem:
            solverOptions["gamma_implicit"] = problem["gamma_implicit"]

        solverOptions["tau_implicit"] = 0.
        if "tau_implicit" in problem:
            solverOptions["tau_implicit"] = problem["tau_implicit"]

        solverOptions["alpha_rayleigh"] = 0.
        if "alpha_rayleigh" in problem:
            solverOptions["alpha_rayleigh"] = problem["alpha_rayleigh"]

        solverOptions["beta_rayleigh"] = 0.05
        if "beta_rayleigh" in problem:
            solverOptions["beta_rayleigh"] = problem["beta_rayleigh"]

        solverOptions["dt_implicit"] = 1.
        if "dt_implicit" in problem:
            solverOptions["dt_implicit"] = problem["dt_implicit"]

        solverOptions["mass_implicit"] = "normal"
        if "mass_implicit" in problem:
            solverOptions["mass_implicit"] = problem["mass_implicit"]

        solverOptions["time_implicit"] = 1.
        if "time_implicit" in problem:
            solverOptions["time_implicit"] = problem["time_implicit"]

        solverOptions["number_of_steps_implicit"] = 500
        if "number_of_steps_implicit" in problem:
            solverOptions["number_of_steps_implicit"] = problem["number_of_steps_implicit"]

        solverOptions["order_implicit"] = 2
        if "order_implicit" in problem:
            solverOptions["order_implicit"] = problem["order_implicit"]

        # Wrinkling model
        solverOptions["sigma_max"] = -1E-6
        solverOptions["iter_goal"] = 5
        solverOptions["breaker"] = 1
        solverOptions["wrinkling_model"] = "none"

        # KDR relaxation
        if "u_relaxation" in problem:
            solverOptions["u_relaxation"] = problem["u_relaxation"]
        else:
            solverOptions["u_relaxation"] = 1.

        if "follower_pressure" in problem:
            solverOptions["follower_pressure"] = problem["follower_pressure"]
        else:
            solverOptions["follower_pressure"] = False

        if "wrinkling_model" in problem:
            solverOptions["wrinkling_model"] = problem["wrinkling_model"]
        else:
            solverOptions["wrinkling_model"] = "none"

        if not "breaker_0" in problem:
            solverOptions["breaker_0"] = 1
        else:
            solverOptions["breaker_0"] = problem["breaker_0"]

        if not "breaker_n" in problem:
            solverOptions["breaker_n"] = 1
        else:
            solverOptions["breaker_n"] = problem["breaker_n"]

        if "sigma_max" in problem:
            solverOptions["sigma_max"] = problem["sigma_max"]

        if "iter_goal" in problem:
            solverOptions["iter_goal"] = problem["iter_goal"]

        # Kinetic dynamic relaxation force convergence criterion
        if not "epsilon_R" in problem:
            solverOptions["epsilon_R"] = 1E-5
        else:
            solverOptions["epsilon_R"] = problem["epsilon_R"]

        # Kinetic dynamic relaxation kinetic energy convergence criterion
        if not "epsilon_KE" in problem:
            solverOptions["epsilon_KE"] = 1E-10
        else:
            solverOptions["epsilon_KE"] = problem["epsilon_KE"]

        # alphaConstant, if True use constant time step
        if not "alphaConstant" in problem:
            # print("alphaConstant not specified. Using default: False")
            solverOptions["alphaConstant"] = 0
        else:
            if problem["alphaConstant"] is False:
                solverOptions["alphaConstant"] = 0
            elif problem["alphaConstant"] is True:
                solverOptions["alphaConstant"] = 1

        # centering parameter for deRooij interior point method
        solverOptions["beta_2"] = 0.1
        if "beta_2" in problem:
            solverOptions["beta_2"] = problem["beta_2"]

        # maximum number of DR outer iterations
        if not "maxIterDR" in problem:
            solverOptions["maxIterDR"] = 1E5
        else:
            solverOptions["maxIterDR"] = problem["maxIterDR"]

        # Insert dicts into object attributes
        self.props = props
        self.load = load
        self.solverOptions = solverOptions

        self.n_homogeneous_BC = 0

        # save covergence for plotting
        self.RoverRF = []
        self.KEoverIE = []
        self.save_res_KE = []
        self.save_res_R = []

        # pre-stress and pre-strain for cable elements
        self.pre_stress_cable = []
        self.pre_strain_cable = []

        # pre-stress and pre-strain for membrane elements
        self.pre_stress_membrane = []
        self.pre_strain_membrane = []

        # prescribed displacements
        self.pre_u = []
        self.pre_u_dof = []
        self.pre_u_flag = False

        self.dof_homogeneous = []

        # 4,1 vector indicating if pre_stress and/or pre_strain are active
        self.pre_active = []

        # wrinkling model arrays
        self.P = []
        self.alpha_array = []

        # save stress and strain before rotation
        self.E_output = []
        self.S_output = []
        self.S1_output = []

        # save field variable
        self.save_Z = []

        # read msh
        self.readMesh()

    def readMesh(self):

        readMsh(self)

        if self.nelems == 0:
            raise Exception("No elements could be read/found. Please check input file")
        elif self.nelemsCable == 0:
            if self.silent is False:
                print("Finished reading \"{}.msh\"\n"
                  "{} membrane elements with {} degrees of freedom in R{} found".format(self.inputName ,
                                                                                        self.nelemsMem,
                                                                                        self.nnodes * self.dim,
                                                                                        self.dim))
        elif self.nelemsMem == 0:
            if self.silent is False:
                print("Finished reading \"{}.msh\"\n"
                  "{} cable elements with {} degrees of freedom in R{} found".format(self.inputName ,
                                                                                   self.nelemsCable,
                                                                                   self.nnodes * self.dim,
                                                                                   self.dim))
        else:
            if self.silent is False:
                print("Finished reading \"{}.msh\"\n"
                  "{} cable elements and {} membrane elements with {} degrees of freedom in R{} found".format(self.inputName,
                                                                                                            self.nelemsCable,
                                                                                                            self.nelemsMem,
                                                                                                            self.nnodes * self.dim,
                                                                                                            self.dim))


    def solve(self):

        if self.solver == "transient":
            if self.silent is False:
                print("Starting transient dynamic solver")
            solveTransient(self)
        elif self.solver == "KDR":
            if self.silent is False:
                print("Starting kinetic dynamic relaxation solver.")
            solveKDR(self)
        else:
            raise Exception("No solver defined.")

        if self.silent is False:
            print("")
            print("********************")
            print("***** ALL DONE *****")
            print("********************")
            print("")


    # cell centres of current configuration
    def computeCellCentre3D(self, dofListInput):

        cdef:
            unsigned int [:] dofList = dofListInput.astype(np.uintc)
            double [:] X = self.X
            double [:] Y = self.Y
            double [:] Z = self.Z
            int [:, ::1] Nm = self.Nm
            double [:, ::1] cc = np.zeros((len(dofList), 3))

        cellCentre3D(dofList, X, Y, Z, Nm, cc)

        self.cc = cc

    # cell centres of current configuration
    def computeCellCentre2D(self):

        if self.nelemsCable == 0:
            raise Exception("No cable elements defined. Cannot compute cell centres of 2D T3 elements.")

        cdef:
            unsigned int nelems = self.nelems
            double [:] X = self.X
            double [:] Y = self.Y
            int [:, ::1] Nb = self.Nb
            double [:, ::1] cc = np.zeros((nelems, 2))

        cellCentre2D(nelems, X, Y, Nb, cc)

        self.cc = cc


    # method callable in python to write current output
    def writeVTK(self):

        writeVTK(self)


    # method to set all loads to zero
    def set_load_zero(self):

        self.loadedBCNodes = np.array([[-1, 0]], dtype=np.double)
        self.loadedBCEdges = np.array([[-1, 0]], dtype=np.double)
        self.elPressurised = np.array([0], dtype=np.uintc)
        self.nPressurised = 0
        self.elFSI = np.array([0], dtype=np.uintc)
        self.nFSI = 0
        self.p = np.array([0], dtype=np.double)

        self.Sx = np.zeros(len(self.elFSI), dtype=np.double)
        self.Sy = np.zeros(len(self.elFSI), dtype=np.double)
        self.Sz = np.zeros(len(self.elFSI), dtype=np.double)
        self.pFSI = np.zeros(len(self.elFSI), dtype=np.double)