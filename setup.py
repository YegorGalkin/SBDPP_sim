import os
import sys
import sysconfig

from distutils.ccompiler import new_compiler
from distutils.errors import DistutilsPlatformError
from setuptools import Extension, find_packages, setup

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Error: Cython is required.")


def check_compiler_type():
    try:
        compiler = new_compiler()
        return compiler.compiler_type
    except DistutilsPlatformError:
        sys.exit("Error: No valid C++ compiler found.")

def get_compile_args():
    args = []
    platform = sysconfig.get_platform()
    
    if platform.startswith("win"):
        args.append("/std:c++20")
    elif platform.startswith(("linux", "darwin")):
        args.extend(["-std=c++20", "-O3"])
    else:
        raise RuntimeError(f"Unsupported platform: {platform}")
    
    return args

compiler_type = check_compiler_type()
compile_args = get_compile_args()

extensions = [
    Extension(
        name="simulation.SpatialBirthDeath",
        sources=[
            "simulation/SpatialBirthDeathWrapper.pyx",
            "src/SpatialBirthDeath.cpp"
        ],
        language="c++",
        include_dirs=[os.path.abspath("include")],
        extra_compile_args=compile_args,
        extra_link_args=compile_args
    )
]

setup(
    name="spatial_sim",
    version="1.0",
    description="Spatial birth-death point process simulator",
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={"language_level": "3"}),
    packages=find_packages(),
)
