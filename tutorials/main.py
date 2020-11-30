import time
import numpy as np
import sys

import mem4py as m4p

import matplotlib.pyplot as plt
import pstats, cProfile


def main():

    print('Starting Hecky plate simulation')
    henckyPlate()

    print('Starting coarse airbag inflation simulation with coarse mesh')
    airbag("coarse")

    # print('Starting airbag inflation simulation with fine mesh')
    # airbag("fine")

    # print('Starting parachute simulation')
    # parachute()


def henckyPlate():

    inputName = "henky"

    elStruc = {}
    elStruc["fixAll"] = {"set": "BC",
                         "type": "fixAll"}

    elStruc["pMembrane"] = {"set": "ELEMENT",
                            "type": "M",
                            "h": 1,
                            "E": 311488,
                            "density": 1E7,
                            "nu": 0.34,
                            "pressure": 100E3}

    problem = {"dim": 3,
               "msh": inputName,
               "solver": "KDR",
               "method": "Barnes",
               "follower_pressure": True,
               "epsilon_R": 1E-10,
               "epsilon_KE": 1E-20,
               "elStruc": elStruc}

    csm = m4p.Mem4py(problem)

    csm.solve()

    print("max Z = {} mm, literature z_max = 32 - 34.8 mm".format(max(csm.Z) * 1000))
    print("")


def airbag(mesh_quality):
    # mesh defined in mm!!
    # full square airbag Flores with 844mm edge length, applied pressure of 5kPa
    # E = 588 MPa

    elStruc = {}

    # inputName = "airbag_quarter_jarasjarungkiat"
    inputName = "airbag_{}".format(mesh_quality)

    elStruc["fixX"] = {"set": "BC",
                       "type": "fixX"}
    elStruc["fixY"] = {"set": "BC",
                       "type": "fixY"}
    elStruc["fixZ"] = {"set": "BC",
                       "type": "fixZ"}

    elStruc["membranes"] = {"set": "ELEMENT",
                            "type": "MW",
                            "h": 0.6,
                            "E": 588,
                            "density": 1E1,
                            "nu": 0.4,
                            "pressure": -5E-3}

    problem = {"dim": 3,
               "msh": inputName,
               "solver": "KDR",
               "method": "Barnes",
               "lam": 1,
               "follower_pressure": True,
               "wrinkling_model": "Jarasjarungkiat",
               "sigma_max": 1E-10,
               "iter_goal": 500,
               "epsilon_R": 1E-5,
               "epsilon_KE": 1E-10,
               "elStruc": elStruc,
               "maxIterDR": 250}

    csm = m4p.Mem4py(problem)
    
    csm.solve()

    print("")
    print("max(Z) = {}".format(max(abs(np.asarray(csm.Z)))))
    print("From literature: Z = {} mm".format(217))
    print("")


def parachute():

    inputName = "parachute"
    # mesh defined in m
    # full parachute with 16 bridles, internal pressure = 5 Pa
    # EMem = 2.07E8 Pa
    # nu = 0.3
    # rhoMem = 9.61 kg/m3
    # t = 1E-5 m
    # ECable = 2.07E9 Pa
    # areaCable = 1.3E-4 m2

    elStruc = {}
    elStruc["fix"] = {"set": "BC",
                      "type": "fixAll"}

    elStruc["fixCanopy"] = {"set": "BC",
                            "type": "fixXY"}

    elStruc["canopy"] = {"set": "ELEMENT",
                         "type": "MW",
                         "h": 3E-5,
                         "E": 2.07E8,
                         "density": 9.61,
                         "nu": 0.3,
                         "pressure": 5E-2}

    elStruc["bridle"] = {"set": "ELEMENT",
                         "type": "CW",
                         "d": 1.3E-4,
                         "E": 2.07E9,
                         "density": 9.61}

    theta = 90 * np.pi / 180
    g = 9.81

    problem = {"dim": 3,
               "msh": inputName,
               "wrinkling_model": "Jarasjarungkiat",
               "sigma_max": 1E-10,
               "iter_goal": 500,
               "solver": "KDR",
               "maxIterDR": 100,
               "method": "Barnes",
               "follower_pressure": True,
               "lam": 0.9,
               "epsilon_R": 1E-5,
               "epsilon_KE": 1E-10,
               "gravity": False,
               "g": [np.sin(theta) * g, 0, np.cos(theta) * g],
               "elStruc": elStruc}

    csm = m4p.Mem4py(problem)

    csm.solve()


if __name__ == "__main__":

    # cProfile.runctx("main()", globals(), locals(), "main.prof")
    # s = pstats.Stats("main.prof")
    # s.strip_dirs().sort_stats("time").print_stats()
    
    start = time.time()
    main()
    end = time.time()
    print("Simulation took {} seconds.\n".format(end - start))
