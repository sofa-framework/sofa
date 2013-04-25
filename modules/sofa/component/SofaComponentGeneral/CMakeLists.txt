cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initComponentGeneral.h
    )
    
set(SOURCE_FILES

    ../initComponentGeneral.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_COMPONENT_GENERAL" )
set(LINKER_DEPENDENCIES SofaValidation SofaExporter SofaEngine SofaGraphComponent SofaTopologyMapping SofaBoundaryCondition SofaUserInteraction SofaConstraint SofaHaptics SofaDenseSolver ${SOFA_VISUAL_LIB} SofaPreconditioner)

if(SOFA-EXTERNAL_HAVE_CSPARSE)
    list(APPEND LINKER_DEPENDENCIES SofaSparseSolver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
