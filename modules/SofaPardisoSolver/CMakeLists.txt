cmake_minimum_required(VERSION 2.8.12)
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
target_link_libraries(${PROJECT_NAME} SofaSimulationTree SofaMeshCollision SofaBaseVisual)
target_link_libraries(${PROJECT_NAME} /media/hpbui/data1_hdd/dev/pardiso/libpardiso500-GNU481-X86-64.so SofaBaseLinearSolver gfortran blas lapack gomp)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PARDISO_SOLVER")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
