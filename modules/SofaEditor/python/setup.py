# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

## Stop on first error
from Cython.Compiler import Options
Options.fast_fail = True

exts = Extension("SofaEditor.*", ["SofaEditor/*.pyx"],
                 libraries=["SofaEditor"],
                 language="c++",
                 extra_compile_args=["-std=c++11"]
                 )

setup(
    packages = ['SofaEditor'],
    ext_modules = cythonize(exts)
)
