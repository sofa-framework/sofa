include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initImplicitODESolver.h 
    ../odesolver/EulerImplicitSolver.h 
    ../odesolver/StaticSolver.h
    ../odesolver/VariationalSymplecticSolver.h

    )
    
set(SOURCE_FILES

    ../initImplicitODESolver.cpp 
    ../odesolver/EulerImplicitSolver.cpp 
    ../odesolver/StaticSolver.cpp
    ../odesolver/VariationalSymplecticSolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMPLICIT_ODE_SOLVER")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
