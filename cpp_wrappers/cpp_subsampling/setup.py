import sys

import numpy
from setuptools import Extension, setup

# Adding OpenCV to project
# ************************

# Adding sources of the project
# *****************************

SOURCES = ["../cpp_utils/cloud/cloud.cpp",
           "grid_subsampling/grid_subsampling.cpp",
           "wrapper.cpp"]

if sys.platform == "win32":
    extra_compile_args = ["/std:c++14"]
else:
    extra_compile_args = ["-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"]

module = Extension(
    name="grid_subsampling",
    sources=SOURCES,
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(ext_modules=[module])








