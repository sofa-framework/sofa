include(${SOFA_CMAKE_DIR}/preProject.cmake)

find_package(OpenCL REQUIRED)

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
    OpenCLSPHFluidForceField.h
    OpenCLSPHFluidForceField.inl
    OpenCLSpatialGridContainer.h
    OpenCLSpatialGridContainer.inl
    OpenCLSphereForceField.h
    OpenCLSphereForceField.inl
    OpenCLSpringForceField.h
    OpenCLSpringForceField.inl
    OpenCLTypes.h
    OpenCLUniformMass.h
    OpenCLUniformMass.inl
    initSofaOpenCL.h
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
    OpenCLSPHFluidForceField.cpp
    OpenCLSpatialGridContainer.cpp
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
    kernels/OpenCLSPHFluidForceField.cl
    kernels/OpenCLSpatialGridContainer.cl
    kernels/OpenCLSpatialGridForce.cl
    kernels/OpenCLSpringForceField.cl
    kernels/OpenCLUniformMass.cl
    kernels/oclRadixSort/RadixSort.cl
    kernels/oclRadixSort/Scan_b.cl
)

include_directories(${OPENCL_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${OTHER_FILES})

AddCompilerDefinitions("SOFA_BUILD_GPU_OPENCL")
AddLinkerDependencies(${OPENCL_LIBRARIES} SofaHelper SofaEngine SofaSphFluid SofaUserInteraction csparse) # taucs taucs_mt system-taucs)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
