include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initImplicitODESolver.h
    EulerImplicitSolver.h
    StaticSolver.h
    VariationalSymplecticSolver.h
    )

set(SOURCE_FILES

    initImplicitODESolver.cpp
    EulerImplicitSolver.cpp
    StaticSolver.cpp
    VariationalSymplecticSolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMPLICIT_ODE_SOLVER")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
