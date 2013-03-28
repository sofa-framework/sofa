cmake_minimum_required(VERSION 2.8)

set(HEADER_FILES

    OpenCLUniformMass.h 
	OpenCLUniformMass.inl

    )

set(SOURCE_FILES

    OpenCLUniformMass.cpp

    )
    
set(OTHER_FILES

    kernels/OpenCLUniformMass.cl

    )    