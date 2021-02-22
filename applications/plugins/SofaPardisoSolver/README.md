README.md
=========

This plugin requires the following dependencies:
- gfortran
- blas
- lapack
- gomp


In the "extlibs/"" folder, please download the pardiso500-GNU461-X86-64 library.

In your home directory, do not forget to create a file "pardiso.lic" which includes the activation key.

When configuring SOFA in CMake, fill the path variable "PARDISO_LIB" with the path to the pardiso library you downloaded in extlibs/.