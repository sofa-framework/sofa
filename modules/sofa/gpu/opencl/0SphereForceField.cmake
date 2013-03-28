cmake_minimum_required(VERSION 2.8)

set(HEADER_FILES

    OpenCLSphereForceField.h 
	OpenCLSphereForceField.inl

    )

set(SOURCE_FILES

    OpenCLSphereForceField.cpp

    )
    
set(OTHER_FILES

    kernels/OpenCLGenericParticleForceField.cl 
	kernels/OpenCLGenericParticleForceField_Sphere.macrocl 

    )    