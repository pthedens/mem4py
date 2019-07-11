# mem4py - A membrane finite element solver based on kinetic dynamic relaxation

This tool provides a solution method to solve the defomations of pressurized membrane structures.
The steady-state is found with the method of kinetic dynamic relaxation which proved its effectivness
for a variety of membrane structures.

# Installation:

mem4py is written in cython which is a C-compilable python script. In benefits from C like speedup
and can be called using a simple python script.

1. Get eigen3 either with

```
sudo apt-get install libeigen3-dev
```

or download under http://eigen.tuxfamily.org.

2. Clone mem4py to folder of choice

```
git clone https://github.com/pthedens/mem4py.git
```

3. If eigen3 is not located at `/usr/include/eigen3` change path to eigen3 in `setup.py` under `include_dirs=[]`.

4. Compile in mem4py/ with

```
"python setup.py"
```

# Test

Run `main.py` to test example cases.

# Setting up your own problem

You can set up your own problem using gmsh and a python script. In gmsh you define the geometry and boundary conditions,
and the python script is used for material and solver properties.

# Mesh:
A surface mesh in gmsh format has to be provided in the /msh folder. 

- Membrane elements are defined by a physical surface named "membrane".

- Boundary conditions are defined on physical lines or points using
  - "fixAll", "fixX", "fixY", "fixZ", "fixXY" and so on to constrain d.o.f. movement
  - nodal loads [N] "fX", "fY", "fZ"
  - edge loads [N/m] "edgeX", "edgeY", "edgeZ"
  - pressure acting on membrane "pMembrane"

# Material and solver properties:

In a python script the properties are defined in a `dict` which is provided to the solver. Closely follow the example
case in `test.py`.

# Acknowledgments

mem4py extensively uses a cython version of eigen3 which is originally based on https://github.com/strohel/Ceygen.
Many thanks to the developers to make eigen3 cython compatible.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 642682 (AWESCO).
