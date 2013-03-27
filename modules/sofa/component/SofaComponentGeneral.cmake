cmake_minimum_required(VERSION 2.8)

project("SofaComponentGeneral")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initComponentGeneral.h

    )
    
set(SOURCE_FILES

    initComponentGeneral.cpp
 
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_COMPONENT_GENERAL" )
set(LINKER_DEPENDENCIES SofaValidation SofaExporter SofaEngine SofaGraphComponent SofaTopologyMapping SofaBoundaryCondition SofaUserInteraction SofaConstraint SofaHaptics SofaDenseSolver SofaPreconditioner SofaOpenglVisual )
# SofaSparseSolver dependency commented out

include(${SOFA_CMAKE_DIR}/post.cmake)
