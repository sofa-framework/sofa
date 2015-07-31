include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initExplicitODESolver.h
    CentralDifferenceSolver.h
    EulerSolver.h
    RungeKutta2Solver.h
    RungeKutta4Solver.h

    )

set(SOURCE_FILES

    initExplicitODESolver.cpp
    CentralDifferenceSolver.cpp
    EulerSolver.cpp
    RungeKutta2Solver.cpp
    RungeKutta4Solver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_EXPLICIT_ODE_SOLVER")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
