cmake_minimum_required(VERSION 2.8)

list(APPEND HEADER_FILES "OpenCLSPHFluidForceField.h")
list(APPEND HEADER_FILES "OpenCLSPHFluidForceField.inl")
list(APPEND HEADER_FILES "CPUSPHFluidForceField.h")
list(APPEND HEADER_FILES "CPUSPHFluidForceFieldWithOpenCL.h")

list(APPEND SOURCE_FILES "OpenCLSPHFluidForceField.cpp")
list(APPEND SOURCE_FILES "CPUSPHFluidForceField.cpp")
list(APPEND SOURCE_FILES "CPUSPHFluidForceFieldWithOpenCL.cpp")

list(APPEND OTHER_FILES "kernels/OpenCLSPHFluidForceField.cl")
