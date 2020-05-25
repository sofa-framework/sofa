cmake_minimum_required(VERSION 3.1)
project(SofaImplicitOdeSolver)

set(HEADER_FILES
    EulerImplicitSolver.h
    StaticSolver.h
    config.h
    initImplicitODESolver.h
)

set(SOURCE_FILES
    EulerImplicitSolver.cpp
    StaticSolver.cpp
    initImplicitODESolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)

sofa_install_targets(SofaCommon ${PROJECT_NAME} "SofaCommon/${PROJECT_NAME}")
