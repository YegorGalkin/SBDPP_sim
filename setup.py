from setuptools import setup, Extension
from distutils.ccompiler import new_compiler
from distutils.errors import DistutilsPlatformError
from Cython.Build import cythonize
import sys


def check_compiler_type():
    try:
        compiler = new_compiler()
        return compiler.compiler_type
    except DistutilsPlatformError:
        sys.exit("Error: No valid C++ compiler found.")


compiler_type = check_compiler_type()
if compiler_type == "msvc":
    compile_args = ["/std:c++17"]
else:
    compile_args = ["-std=c++17"]

extensions = [
    Extension(
        # 1) Remove dot to avoid mismatch in init symbol
        name="simulation.SpatialBirthDeath",
        sources=["simulation/SpatialBirthDeathWrapper.pyx",
                 "simulation/SpatialBirthDeath.cpp"],
        language="c++",
        include_dirs=["include"],
        extra_compile_args=compile_args
    )
]

setup(
    name="spatial_sim",
    version="1.0",
    description="Spatial birth-death point process simulator",
    ext_modules=cythonize(extensions, annotate=True,
                          compiler_directives={"language_level": "3"}),
    packages=["SpatialBirthDeath"],
)