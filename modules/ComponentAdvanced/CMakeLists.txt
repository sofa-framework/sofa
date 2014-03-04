include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initComponentAdvanced.h)

set(SOURCE_FILES
    initComponentAdvanced.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_ADVANCED")
AddLinkerDependencies(EulerianFluid SphFluid VolumetricData NonUniformFem)
if(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)
    AddLinkerDependencies(Eigen2Solver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

