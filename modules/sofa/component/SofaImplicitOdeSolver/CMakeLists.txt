include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initImplicitODESolver.h 
    ../odesolver/EulerImplicitSolver.h 
    ../odesolver/StaticSolver.h

    )
    
set(SOURCE_FILES

    ../initImplicitODESolver.cpp 
    ../odesolver/EulerImplicitSolver.cpp 
    ../odesolver/StaticSolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_IMPLICIT_ODE_SOLVER")
set(LINKER_DEPENDENCIES SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
