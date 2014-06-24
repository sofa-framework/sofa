include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initComponentGeneral.h
    )

set(SOURCE_FILES

    initComponentGeneral.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_COMPONENT_GENERAL" )
AddLinkerDependencies(SofaValidation SofaExporter SofaEngine SofaGraphComponent SofaTopologyMapping SofaBoundaryCondition SofaUserInteraction SofaConstraint SofaHaptics SofaDenseSolver ${SOFA_VISUAL_LIB} SofaPreconditioner)

if(SOFA-EXTERNAL_CSPARSE)
    AddLinkerDependencies(SofaSparseSolver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
