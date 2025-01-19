from setuptools import setup, Extension
from distutils.ccompiler import new_compiler
from distutils.errors import DistutilsPlatformError
from Cython.Build import cythonize
import sys
import platform


def read_requirements():
    """
    Reads the requirements.txt file and returns a list of dependencies.
    """
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def check_compiler_type():
    """
    Checks and prints the compiler type detected by setuptools.
    Raises an error if no compiler is found.
    """
    try:
        compiler = new_compiler()
        compiler_type = compiler.compiler_type
        print(f"Detected compiler: {compiler_type}")
        return compiler_type
    except DistutilsPlatformError as e:
        sys.stderr.write("Error: Could not detect a valid C++ compiler.\n")
        sys.stderr.write("Ensure you have a C++ compiler installed and properly configured.\n")
        sys.stderr.write("For Windows, install Visual Studio with C++ Build Tools.\n")
        sys.stderr.write("For Linux/Mac, ensure g++ is installed and accessible via PATH.\n")
        raise RuntimeError("C++ compiler not detected") from e


# Detect and print compiler type, or exit on failure
try:
    compiler_type = check_compiler_type()
except RuntimeError as e:
    sys.exit(1)

extensions = [
    Extension(
        "simulation.grid_1d_interface",  # Python module name
        sources=["simulation/grid_1d_interface.pyx", "simulation/grid_1d.cpp"],  # Source files
        include_dirs=["include"],  # Include directory for header files
        language="c++",
        extra_compile_args=[
            "-O3",  # Maximum optimization
            "-march=native",  # Use the architecture of the current machine
            "-ffast-math",  # Enable fast math optimizations
            "-funroll-loops",  # Unroll loops
        ] if platform.system() != "Windows" else [
            "/O2",  # Full optimization for MSVC
            "/fp:fast",  # Fast floating-point model
            "/arch:AVX2",  # Use AVX2 instruction set (if supported by the CPU)
        ],
    )
]

setup(
    name="grid_1d",
    version="1.0",
    description="Spatial birth-death point process simulator.",
    author="Egor Galkin",
    author_email="",
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            "language_level": "3",  # Python 3
            "boundscheck": False,  # Disable bounds checking
            "wraparound": False,  # Disable negative indexing checks
            "cdivision": True,  # Optimize division
            "profile": False,  # Disable profiling
        },
    ),
    install_requires=read_requirements(),
    packages=["grid_1d"],
    zip_safe=False,
)