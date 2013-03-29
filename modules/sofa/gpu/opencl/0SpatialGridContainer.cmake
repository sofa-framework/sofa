cmake_minimum_required(VERSION 2.8)

list(APPEND HEADER_FILES "OpenCLSpatialGridContainer.h")
list(APPEND HEADER_FILES "OpenCLSpatialGridContainer.inl")

list(APPEND SOURCE_FILES "OpenCLSpatialGridContainer.cpp")

list(APPEND OTHER_FILES "kernels/OpenCLSpatialGridContainer.cl")
