list(APPEND HEADER_FILES "OpenCLSphereForceField.h")
list(APPEND HEADER_FILES "OpenCLSphereForceField.inl")

list(APPEND SOURCE_FILES "OpenCLSphereForceField.cpp")

list(APPEND OTHER_FILES "kernels/OpenCLGenericParticleForceField.cl")
list(APPEND OTHER_FILES "kernels/OpenCLGenericParticleForceField_Sphere.macrocl")
