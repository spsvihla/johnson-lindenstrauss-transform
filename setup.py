#!/usr/bin/env python

from setuptools import setup, Extension, find_packages
from glob import glob
import numpy as np

PKG="jlt"
SRC="src/"

ext_mod = Extension(
    name=PKG, #+"."+cmod,
    sources=glob(SRC+"*.c"),
    include_dirs=[
        '/usr/include/x86_64-linux-gnu',    # BLAS
        np.get_include()],                  # NumPy
    library_dirs=[
        '/usr/lib/x86_64-linux-gnu/blas'    # BLAS
    ],
    libraries=[
        'cblas'
    ],
    extra_link_args=['-lm'] # link math.h
)

setup(
    name=PKG,
    url="https://github.com/spsvihla/algorithms-toolbox",
    version='0.1',
    description='A collection of algorithms in Python',
    author='Sean Svihla',
    setup_requires=["numpy"],
    install_requires=[
        "numpy"
    ],
    packages=find_packages(where=SRC),
    ext_modules=[ext_mod],
    package_dir={"": SRC},
    include_package_data=True
)