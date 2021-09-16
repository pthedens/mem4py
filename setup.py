import os
import numpy
from glob import glob
from distutils.core import setup
from setuptools import find_packages, Extension
#from distutils.extension import Extension
from pathlib import Path

useCython = True

APPNAME = "mem4py"
APPVERSION = "1.0.1"

if os.name == 'nt':
    EIGEN_PATH = Path(r'C:\Program Files\eigen3')
elif os.name == 'posix':
    EIGEN_PATH = Path('/usr/include/eigen3')

cmdclass = dict()
if useCython:
    try:
        from Cython.Distutils import build_ext
        cmdclass.update({'build_ext': build_ext})
    except ImportError:
        print("WARNING: No Cython installed, compiling from .cpp files")
        useCython = False

ext = '.pyx' if useCython else '.cpp'
files = glob('mem4py/**/*{}'.format(ext), recursive=True)


def makeExtension(file, includeDirs):
    # There probably is a better way of doing this..
    # Extension namespace.package.name
    nameSpace = os.path.relpath(file, Path('mem4py/'))
    extName = os.path.splitext(nameSpace)[0].replace(os.path.sep, '.')
    extName = APPNAME if (extName == f"{APPNAME}.{APPNAME}") else extName  # Handle mem4py.mem4py
    return Extension(extName, [file], include_dirs=includeDirs, language='c++')


includeDirs = list(set([os.path.dirname(path) for path in files]))
includeDirs.append(numpy.get_include())
includeDirs.append(EIGEN_PATH)
extensions = [makeExtension(path, includeDirs) for path in files]

setup(
    name=APPNAME,
    version=APPVERSION,
    url="https://github.com/pthedens/mem4py",
    description="Python interface for membrane FEM solver.",
    ext_modules=extensions,
    cmdclass=cmdclass,
    package_dir={'': 'mem4py'},
    packages=find_packages(where='mem4py'),
    # packages=['mem4py', 'mem4py/assembler', 'mem4py/ceygen',
    #           'mem4py/solver', 'mem4py/helper', 'mem4py/elements']
)

print("********CYTHON COMPLETE******")
