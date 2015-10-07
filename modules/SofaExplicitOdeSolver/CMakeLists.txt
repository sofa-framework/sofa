cmake_minimum_required(VERSION 2.8.12)
project(SofaExplicitOdeSolver)

set(HEADER_FILES
    CentralDifferenceSolver.h
    EulerSolver.h
    RungeKutta2Solver.h
    RungeKutta4Solver.h
    config.h
    initExplicitODESolver.h
)

set(SOURCE_FILES
    CentralDifferenceSolver.cpp
    EulerSolver.cpp
    RungeKutta2Solver.cpp
    RungeKutta4Solver.cpp
    initExplicitODESolver.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_EXPLICIT_ODE_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
