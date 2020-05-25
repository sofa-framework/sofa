cmake_minimum_required(VERSION 3.1)
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

sofa_install_targets(SofaGeneral ${PROJECT_NAME} "SofaGeneral/${PROJECT_NAME}")
