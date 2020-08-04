cmake_minimum_required(VERSION 3.12)
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

sofa_add_targets_to_package(
    PACKAGE_NAME SofaCommon
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
