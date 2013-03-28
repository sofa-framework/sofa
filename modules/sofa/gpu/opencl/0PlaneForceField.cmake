cmake_minimum_required(VERSION 2.8)

set(HEADER_FILES

    OpenCLPlaneForceField.h 
	OpenCLPlaneForceField.inl
	
    )

set(SOURCE_FILES

    OpenCLPlaneForceField.cpp

    )
    
set(OTHER_FILES

    kernels/OpenCLGenericParticleForceField.cl 
	kernels/OpenCLGenericParticleForceField_Plane.macrocl

)