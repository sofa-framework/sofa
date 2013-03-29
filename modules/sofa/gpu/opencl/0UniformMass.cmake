cmake_minimum_required(VERSION 2.8)


list(APPEND HEADER_FILES "OpenCLUniformMass.h")
list(APPEND HEADER_FILES "OpenCLUniformMass.inl")

list(APPEND SOURCE_FILES "OpenCLUniformMass.cpp")

list(APPEND OTHER_FILES "kernels/OpenCLUniformMass.cl")
