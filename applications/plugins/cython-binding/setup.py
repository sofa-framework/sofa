from distutils.core import setup, Extension
import pkg_resources
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os

sofaroot  = os.environ['SOFAROOTDIR'] 
sofabuild = os.environ['SOFABUILDDIR']

data_dir = pkg_resources.resource_filename("autowrap", "data_files")
sofa_dir = sofaroot+"/framework"
sofamod_dir = sofaroot+"/modules"

sofaroot_dir = sofaroot
sofabuild_dir = sofabuild+"/include"
sofaext_dir = sofaroot+"/extlibs/eigen-3.2.7/"
sofaxml_dir = sofaroot+"/extlibs/tinyxml/"

ext = Extension("sofacython", sources = ['src/pyx/sofacython.pyx'], language="c++",
        extra_compile_args = [],
        include_dirs = ["./src/pxd", ".", "../", data_dir, sofaroot_dir, sofa_dir, sofabuild_dir, sofaext_dir, sofamod_dir, sofaxml_dir],
        extra_link_args = [],
        )  

setup(cmdclass = {'build_ext' : build_ext},
    name="sofacython",
    packages=["sofacython"],
    version="0.0.1",
    ext_modules = [ext]
    )
