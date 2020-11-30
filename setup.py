import os
import numpy
from glob import glob
from distutils.core import setup
from distutils.extension import Extension

useCython = True

APPNAME = "mem4py"
APPVERSION = "1.0"

cmdclass = dict()
if useCython:
    try:
        from Cython.Distutils import build_ext
        cmdclass.update({'build_ext': build_ext})
    except ImportError:
        print("WARNIG: No Cython installed, compiling from .cpp files")
        useCython = False

ext = '.pyx' if useCython else '.cpp'


files = glob('src/*{}'.format(ext))
files.extend(glob('src/*/*{}'.format(ext)))

def makeExtensionLinux(file, includeDird):
    extName = os.path.splitext(file)[0]
    extName = extName.replace(os.path.sep, '.')
    return Extension(extName, [file], include_dirs = includeDirs, language='c++')

def makeExtensionWindows(file, includeDirs):
    extName = file.replace('\\', '.')[:-4]
    return Extension(extName, [file], include_dirs=includeDirs, language='c++')

includeDirs = list(set([os.path.dirname(path) for path in files]))
if os.name == 'nt':
    includeDirs.append('C:\\Program Files\\eigen3')
    includeDirs.append(numpy.get_include())
    extensions = [makeExtensionWindows(path, includeDirs) for path in files]
elif os.name == 'posix':
    includeDirs.append('/usr/include/eigen3')
    includeDirs.append(numpy.get_include())
    extensions = [makeExtensionLinux(path, includeDirs) for path in files]

setup(
  name=APPNAME,
  version=APPVERSION,
  description="Python interface for membrane FEM solver.",
  ext_modules=extensions,
  cmdclass=cmdclass,
  packages=['src', 'src/assembler', 'src/ceygen',
            'src/solver', 'src/helper', 'src/elements']
)

print("********CYTHON COMPLETE******")
