cmake_minimum_required(VERSION 3.12)
project(SofaGeneralImplicitOdeSolver)

set(HEADER_FILES
    config.h
    initGeneralImplicitODESolver.h
    )

set(SOURCE_FILES
    initGeneralImplicitODESolver.cpp
    )

list(APPEND HEADER_FILES
    VariationalSymplecticSolver.h
    )
list(APPEND SOURCE_FILES
    VariationalSymplecticSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaGeneral
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
