cmake_minimum_required(VERSION 2.8)

set(HEADER_FILES

    OpenCLSpatialGridContainer.h 
	OpenCLSpatialGridContainer.inl

    )

set(SOURCE_FILES

    OpenCLSpatialGridContainer.cpp

    )
    
set(OTHER_FILES

    kernels/OpenCLSpatialGridContainer.cl

    )    