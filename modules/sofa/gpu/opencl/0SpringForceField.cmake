cmake_minimum_required(VERSION 2.8)

set(HEADER_FILES

    OpenCLSpringForceField.h 
	OpenCLSpringForceField.inl

    )

set(SOURCE_FILES

    OpenCLSpringForceField.cpp

    )
    
set(OTHER_FILES

    kernels/OpenCLSpringForceField.cl

    )    