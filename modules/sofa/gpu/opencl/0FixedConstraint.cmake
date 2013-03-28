cmake_minimum_required(VERSION 2.8)

set(HEADER_FILES

    OpenCLFixedConstraint.h 
	OpenCLFixedConstraint.inl
    
    )

set(SOURCE_FILES

    OpenCLFixedConstraint.cpp
    
    )
    
set(OTHER_FILES

    kernels/OpenCLFixedConstraint.cl
    
    )