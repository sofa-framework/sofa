cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initComponentBase.h

    )
    
set(SOURCE_FILES

    ../initComponentBase.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_COMPONENT_BASE")
set(LINKER_DEPENDENCIES SofaBaseTopology SofaBaseMechanics SofaBaseCollision SofaBaseLinearSolver SofaBaseAnimationLoop SofaBaseVisual)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
