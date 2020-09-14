cmake_minimum_required(VERSION 3.12)
project(SofaExplicitOdeSolver)

set(HEADER_FILES
    EulerSolver.h
    config.h
    initExplicitODESolver.h
)

set(SOURCE_FILES
    EulerSolver.cpp
    initExplicitODESolver.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaCommon
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
