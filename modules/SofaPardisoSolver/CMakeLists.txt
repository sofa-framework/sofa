cmake_minimum_required(VERSION 2.8)
project(SofaPardisoSolver)

set(HEADER_FILES
    SparsePARDISOSolver.h
    config.h
)

set(SOURCE_FILES
    SparsePARDISOSolver.cpp
    initPardisoSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

add_definitions("SOFA_BUILD_PARDISO_SOLVER")
target_link_libraries(${PROJECT_NAME} SofaSimulationTree SofaMeshCollision SofaBaseVisual)
target_link_libraries(${PROJECT_NAME} pardiso500-GNU461-X86-64 gfortran blas lapack gomp)
