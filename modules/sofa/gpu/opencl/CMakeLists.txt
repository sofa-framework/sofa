include(${SOFA_CMAKE_DIR}/preProject.cmake)

find_package(OpenCL REQUIRED)

set(HEADER_FILES

	myopencl.h
    OpenCLKernel.h
    OpenCLProgram.h
    OpenCLProgramParser.h
    OpenCLTypes.h
    gpuopencl.h
    OpenCLCommon.h
	
	OpenCLSphereForceField.h
	OpenCLSphereForceField.inl
	
	OpenCLSPHFluidForceField.h
    OpenCLSPHFluidForceField.inl
    CPUSPHFluidForceField.h
    CPUSPHFluidForceFieldWithOpenCL.h
	
	OpenCLSpringForceField.h
	OpenCLSpringForceField.inl
	
	OpenCLUniformMass.h
	OpenCLUniformMass.inl
	
	OpenCLFixedConstraint.h
	OpenCLFixedConstraint.inl
	
	OpenCLIdentityMapping.h
	OpenCLIdentityMapping.inl
	
	OpenCLMechanicalObject.h
	OpenCLMechanicalObject.inl
	
	OpenCLMemoryManager.h
	
	OpenCLPlaneForceField.h
	OpenCLPlaneForceField.inl
	
	OpenCLSpatialGridContainer.h
	OpenCLSpatialGridContainer.inl
	
	oclRadixSort/RadixSort.h
    oclRadixSort/Scan.h
    oclRadixSort/CPUSortWithOpenCL.h
	oclRadixSort/CPUSortWithCuda.h
	
	tools/top.h
	tools/showvector.h
	)

set(SOURCE_FILES

	myopencl.cpp
    OpenCLCommon.cpp
    OpenCLProgram.cpp
    OpenCLMouseInteraction.cpp
	
	OpenCLSphereForceField.cpp
	
	OpenCLSPHFluidForceField.cpp
    CPUSPHFluidForceField.cpp
    CPUSPHFluidForceFieldWithOpenCL.cpp
	
	OpenCLSpringForceField.cpp
	
	OpenCLUniformMass.cpp
	
	OpenCLBoxROI.cpp
	
	OpenCLFixedConstraint.cpp
	
	OpenCLIdentityMapping.cpp
	
	OpenCLMechanicalObject.cpp
	
	OpenCLMemoryManager.cpp
	
	OpenCLPlaneForceField.cpp
	
	OpenCLSpatialGridContainer.cpp
	
	oclRadixSort/RadixSort.cpp
    oclRadixSort/Scan.cpp
	
	tools/top.cpp
	tools/showvector.cpp
	)
	
set(OTHER_FILES

	kernels/OpenCLGenericParticleForceField.cl
	
	kernels/OpenCLSPHFluidForceField.cl
	
	kernels/OpenCLSpringForceField.cl
	
	kernels/OpenCLUniformMass.cl
	
	kernels/OpenCLFixedConstraint.cl
	
	kernels/OpenCLMechanicalObject.cl
	
	kernels/OpenCLMemoryManager.cl
	
	kernels/OpenCLGenericParticleForceField_Plane.macrocl
	
	kernels/OpenCLGenericParticleForceField_Sphere.macrocl
	
	kernels/OpenCLSpatialGridContainer.cl

    kernels/OpenCLSpatialGridForce.cl
	
	kernels/oclRadixSort/RadixSort.cl
    kernels/oclRadixSort/Scan_b.cl
	)

include_directories(${OPENCL_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${OTHER_FILES})

AddCompilerDefinitions("SOFA_BUILD_GPU_OPENCL")
AddLinkerDependencies(${OPENCL_LIBRARIES} SofaHelper SofaEngine SofaSphFluid SofaUserInteraction csparse) # taucs taucs_mt system-taucs)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
