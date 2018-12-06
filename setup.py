import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = ['.'] #your include_dirs must contains the '.' for setup to search all the subfolder of the codeRootFolder
        )

extNames = scandir('src')

extensions = [makeExtension(name) for name in extNames]
for module in extensions:
    module.language = "c++"

setup(
  name="mem4py",
  ext_modules=extensions,
  include_dirs=['/usr/include/eigen3'],  # default overridable by setup.cfg
  language= "c++",
  cmdclass={'build_ext': build_ext},
  script_args=['build_ext'],
  options={'build_ext': {'inplace': True}}
)

print("********CYTHON COMPLETE******")
