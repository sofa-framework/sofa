cmake_minimum_required(VERSION 2.8)

list(APPEND HEADER_FILES "OpenCLPlaneForceField.h")
list(APPEND HEADER_FILES "OpenCLPlaneForceField.inl")

list(APPEND SOURCE_FILES "OpenCLPlaneForceField.cpp")

list(APPEND OTHER_FILES "kernels/OpenCLGenericParticleForceField.cl")
list(APPEND OTHER_FILES "kernels/OpenCLGenericParticleForceField_Plane.macrocl")
