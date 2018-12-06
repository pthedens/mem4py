import time
import numpy as np

import src.mem4py as m4p

import matplotlib.pyplot as plt
import pstats, cProfile

def main():

    ##############################################
    ### BENCHMARK TESTS FOR MEMBRANE ANALYSIS ####
    ##############################################

    inputName = "2DplateWithHole"
    # benchmark for 2D static analysis:
    # quarter plate with hole (R=0.5m) and applied load of 10kPa
    # stress at hole should converge to 30kPa (stress concentration factor k=3)
    problem = {"EMem": 2E11,
               "nu": 0.3,
               "dim": 2,
               "msh": inputName,
               "NL": False,
               "epsilonNR": 1E-5,
               "mat": "SVIso",
               "t": 1,
               "edgeX": 10E3}

    csm = m4p.Mem4py(problem)
    csm.solve()

    inputName = "mh92symmetry"
    V = 40
    problem = {"EMem": 5E4,
                   "nu": 0.3,
                   "dim": 3,
                   "msh": inputName,
                   "wrinkling": True,
                   "DR": True,
                   "epsilonR": 1E-5,
                   "epsilonKE": 1E-8,
                   "mat": "SVIso",
                   "t": 1,
                   "p": 1.225 * 0.5 * V ** 2}

    csm = m4p.Mem4py(problem)
    csm.solve()

    inputName = "henky"
    # henky plate (z_max = 32 - 34.8 mm)
    problem = {"EMem": 311488,
               "nu": 0.34,
               "dim": 3,
               "msh": inputName,
               "wrinkling": True,
               "DR": True,
               "epsilonR": 1E-5,
               "epsilonKE": 1E-10,
               "mat": "SVIso",
               "t": 1,
               "p": 100E3}

    csm = m4p.Mem4py(problem)
    csm.solve()
    print("maximum Z deflection = ", max(csm.Z))


if __name__ == "__main__":

    # cProfile.runctx("main()", globals(), locals(), "main.prof")
    # s = pstats.Stats("main.prof")
    # s.strip_dirs().sort_stats("time").print_stats()

    start = time.time()
    main()
    end = time.time()
    print("Simulation took {} seconds.\n".format(end - start))
