cmake_minimum_required(VERSION 3.1)
project(SofaGeneralExplicitOdeSolver)

set(HEADER_FILES
    CentralDifferenceSolver.h
    RungeKutta2Solver.h
    RungeKutta4Solver.h
    config.h
    initGeneralExplicitODESolver.h
)

set(SOURCE_FILES
    CentralDifferenceSolver.cpp
    RungeKutta2Solver.cpp
    RungeKutta4Solver.cpp
    initGeneralExplicitODESolver.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_EXPLICIT_ODE_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
