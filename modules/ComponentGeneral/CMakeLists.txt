include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initComponentGeneral.h)

set(SOURCE_FILES
    initComponentGeneral.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_GENERAL")
AddLinkerDependencies(Validation Exporter Engine GraphComponent TopologyMapping BoundaryCondition UserInteraction Constraint Haptics DenseSolver ${SOFA_VISUAL_LIB} Preconditioner)

if(SOFA-EXTERNAL_CSPARSE)
    AddLinkerDependencies(SparseSolver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

