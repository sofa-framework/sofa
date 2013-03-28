cmake_minimum_required(VERSION 2.8)

set(OBJECT_DIR "OpenCLMechanicalObject" )

set(HEADER_FILES

    OpenCLMechanicalObject.h 
	OpenCLMechanicalObject.inl
	
    )

set(SOURCE_FILES

    OpenCLMechanicalObject.cpp

    )
    
set(OTHER_FILES

    kernels/OpenCLMechanicalObject.cl

    )