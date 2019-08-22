cmake_minimum_required(VERSION 3.1)
project(SofaOpenCL)

set(SOFAOPENCL_VERSION 0.1)

set(HEADER_FILES
    CPUSPHFluidForceField.h
    CPUSPHFluidForceFieldWithOpenCL.h
    OpenCLCommon.h
    OpenCLFixedConstraint.h
    OpenCLFixedConstraint.inl
    OpenCLIdentityMapping.h
    OpenCLIdentityMapping.inl
    OpenCLKernel.h
    OpenCLMechanicalObject.h
    OpenCLMechanicalObject.inl
    OpenCLMemoryManager.h
    OpenCLPlaneForceField.h
    OpenCLPlaneForceField.inl
    OpenCLProgram.h
    OpenCLProgramParser.h
    OpenCLSphereForceField.h
    OpenCLSphereForceField.inl
    OpenCLSpringForceField.h
    OpenCLSpringForceField.inl
    OpenCLTypes.h
    OpenCLUniformMass.h
    OpenCLUniformMass.inl
    config.h
    myopencl.h
    oclRadixSort/CPUSortWithCuda.h
    oclRadixSort/CPUSortWithOpenCL.h
    oclRadixSort/RadixSort.h
    oclRadixSort/Scan.h
    tools/showvector.h
    tools/top.h
)

set(SOURCE_FILES
    CPUSPHFluidForceField.cpp
    CPUSPHFluidForceFieldWithOpenCL.cpp
    OpenCLBoxROI.cpp
    OpenCLCommon.cpp
    OpenCLFixedConstraint.cpp
    OpenCLIdentityMapping.cpp
    OpenCLMechanicalObject.cpp
    OpenCLMemoryManager.cpp
    OpenCLMouseInteraction.cpp
    OpenCLPlaneForceField.cpp
    OpenCLProgram.cpp
    OpenCLSphereForceField.cpp
    OpenCLSpringForceField.cpp
    OpenCLUniformMass.cpp
    initSofaOpenCL.cpp
    myopencl.cpp
    oclRadixSort/RadixSort.cpp
    oclRadixSort/Scan.cpp
    tools/showvector.cpp
    tools/top.cpp
)

set(OTHER_FILES
    kernels/OpenCLFixedConstraint.cl
    kernels/OpenCLGenericParticleForceField.cl
    kernels/OpenCLGenericParticleForceField_Plane.macrocl
    kernels/OpenCLGenericParticleForceField_Sphere.macrocl
    kernels/OpenCLMechanicalObject.cl
    kernels/OpenCLMemoryManager.cl
    kernels/OpenCLSpatialGridForce.cl
    kernels/OpenCLSpringForceField.cl
    kernels/OpenCLUniformMass.cl
    kernels/oclRadixSort/RadixSort.cl
    kernels/oclRadixSort/Scan_b.cl
)

if(SofaSphFluid_FOUND)
    list(APPEND HEADER_FILES
        OpenCLSPHFluidForceField.h
        OpenCLSPHFluidForceField.inl
        OpenCLSpatialGridContainer.h
        OpenCLSpatialGridContainer.inl)
    list(APPEND SOURCE_FILES
        OpenCLSPHFluidForceField.cpp
        OpenCLSpatialGridContainer.cpp)
    list(APPEND OTHER_FILES
        kernels/OpenCLSPHFluidForceField.cl
        kernels/OpenCLSpatialGridContainer.cl
        )
    message(STATUS "SofaOpenCL: optional dependency to SofaSphFluid found. Activing <OpenCLSPHFluidForceField>.")
else()
    message(STATUS "SofaOpenCL: optional dependency to SofaSphFluid not found. De-activing <OpenCLSPHFluidForceField>")
endif()


find_package(OpenCL REQUIRED)
find_package(SofaGeneral REQUIRED)

message(WARNING "SofaOpenCL: TODO: don't use absolute path (SOFA_SRC_DIR)")

include_directories(${OPENCL_INCLUDE_DIRS})
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${OTHER_FILES})
add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../..\"")

add_definitions("-DSOFA_BUILD_GPU_OPENCL")
target_link_libraries(${PROJECT_NAME} ${OPENCL_LIBRARIES} SofaHelper SofaEngine SofaSphFluid SofaUserInteraction csparse) # taucs taucs_mt system-taucs)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(SofaOpenCL ${SOFAOPENCL_VERSION} SofaOpenCL SofaOpenCL)
