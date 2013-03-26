cmake_minimum_required(VERSION 2.8)

project("SofaPardisoSolver")

include(${SOFA_CMAKE_DIR}/pre.cmake)

if(SOFA_HAVE_PARDISO)
set(HEADER_FILES

    SparsePARDISOSolver.h

    )
set(SOURCE_FILES

    SparsePARDISOSolver.cpp

    )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PARDISO_SOLVER")
set(LINKER_DEPENDENCIES SofaSimulationTree SofaMeshCollision SofaBaseVisual)

include(${SOFA_CMAKE_DIR}/post.cmake)
