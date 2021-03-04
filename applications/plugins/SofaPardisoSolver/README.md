README.md
=========

This plugin requires the following dependencies:
- gfortran
- blas
- lapack
- gomp

Download the Pardiso library and license key from https://www.pardiso-project.org

In your home directory, create a file "pardiso.lic" which includes the license key.

When configuring SOFA in CMake, fill the path variable "PARDISO_LIB" with the path to the pardiso library you downloaded.
