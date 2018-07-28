# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
import sys, os, glob

if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False

ext = 'pyx' if USE_CYTHON else 'cpp'

def toImport(s):
    return s.replace("/",".")[:-4]

sources = glob.iglob("SofaEditor/*."+ext)
exts=[]
for source in sources:
    print("Generating the binding using:" + source)
    exts.append(Extension(toImport(source), [source],
                    libraries=["SofaEditor"],
                    language="c++",
                    extra_compile_args=["-std=c++11"]
                    ))

if USE_CYTHON:
    from Cython.Build import cythonize

    ## Stop on first error
    from Cython.Compiler import Options
    Options.fast_fail = True

    exts = cythonize(exts)

setup(
    packages = ['SofaEditor'],
    ext_modules = exts
)
