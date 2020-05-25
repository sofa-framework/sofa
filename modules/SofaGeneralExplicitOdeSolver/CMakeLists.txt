cmake_minimum_required(VERSION 3.1)
project(SofaGeneralExplicitOdeSolver)

set(HEADER_FILES
    config.h
    initGeneralExplicitODESolver.h
    )

set(SOURCE_FILES
    initGeneralExplicitODESolver.cpp
    )

list(APPEND HEADER_FILES
    CentralDifferenceSolver.h
    RungeKutta2Solver.h
    RungeKutta4Solver.h
    )
list(APPEND SOURCE_FILES
    CentralDifferenceSolver.cpp
    RungeKutta2Solver.cpp
    RungeKutta4Solver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)

sofa_install_targets(SofaGeneral ${PROJECT_NAME} "SofaGeneral/${PROJECT_NAME}")
