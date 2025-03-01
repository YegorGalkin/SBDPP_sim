import os
import sysconfig
from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize


def get_compiler_args():
    compiler = sysconfig.get_config_var("CC")

    compiler_str_lower = "" if compiler is None else compiler.lower()
    if "msvc" in compiler_str_lower:
        # Use /O2 for maximum optimization on MSVC
        compile_args = ["/std:c++20", "/O2"]
        link_args = [] # MSVC typically does not need optimization flags for linking
    elif "gcc" in compiler_str_lower or "clang" in compiler_str_lower:
        # For GCC/Clang, add -O3 and -march=native for maximum optimization
        compile_args = ["-std=c++20", "-O3", "-march=native"]
        link_args = compile_args
    else:
        raise RuntimeError(f"Unsupported compiler: {compiler}. Only MSVC, GCC, and Clang are supported.")
        
    return compile_args, link_args


compile_args, link_args = get_compiler_args()


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
        extra_link_args=link_args
    )
]

setup(
    name="spatial_sim",
    version="1.0",
    description="Spatial birth-death point process simulator",
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={"language_level": "3"}),
    packages=find_packages(),
)
