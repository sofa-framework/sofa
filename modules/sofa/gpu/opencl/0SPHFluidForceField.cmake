cmake_minimum_required(VERSION 2.8)

set(HEADER_FILES

    OpenCLSPHFluidForceField.h 
    OpenCLSPHFluidForceField.inl 
    CPUSPHFluidForceField.h 
    CPUSPHFluidForceFieldWithOpenCL.h

    )

set(SOURCE_FILES

    OpenCLSPHFluidForceField.cpp 
    CPUSPHFluidForceField.cpp 
    CPUSPHFluidForceFieldWithOpenCL.cpp

    )
    
set(OTHER_FILES

    kernels/OpenCLSPHFluidForceField.cl

    )    