include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES  )
set(SOURCE_FILES  )
set(OTHER_FILES  )

include("0BoxROI.cmake")
include("0Core.cmake")
include("0FixedConstraint.cmake")
include("0IdentityMapping.cmake")
include("0MechanicalObject.cmake")
include("0MemoryManager.cmake")
include("0PlaneForceField.cmake")
include("0SpatialGridContainer.cmake")
include("0SphereForceField.cmake")
include("0SPHFluidForceField.cmake")
include("0SpringForceField.cmake")
include("0UniformMass.cmake")

add_subdirectory("kernels")
add_subdirectory("oclRadixSort")
add_subdirectory("tools")
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${OTHER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_GPU_OPENCL")
set(LINKER_DEPENDENCIES SofaHelper SofaEngine SofaSphFluid SofaUserInteraction opencl csparse taucs taucs_mt system-taucs)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
