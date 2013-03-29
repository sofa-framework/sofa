cmake_minimum_required(VERSION 2.8)

list(APPEND HEADER_FILES "OpenCLFixedConstraint.h")
list(APPEND HEADER_FILES "OpenCLFixedConstraint.inl")

list(APPEND SOURCE_FILES "OpenCLFixedConstraint.cpp")

list(APPEND OTHER_FILES "kernels/OpenCLFixedConstraint.cl")
