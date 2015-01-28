include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initExplicitODESolver.h 
    # ../odesolver/CentralDifferenceSolver.h 
    ../odesolver/EulerSolver.h 
    # ../odesolver/RungeKutta2Solver.h 
    # ../odesolver/RungeKutta4Solver.h

    )
    
set(SOURCE_FILES

    ../initExplicitODESolver.cpp 
    # ../odesolver/CentralDifferenceSolver.cpp 
    ../odesolver/EulerSolver.cpp 
    # ../odesolver/RungeKutta2Solver.cpp 
    # ../odesolver/RungeKutta4Solver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_EXPLICIT_ODE_SOLVER")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
