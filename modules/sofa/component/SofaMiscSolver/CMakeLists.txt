include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMiscSolver.h 
    # ../odesolver/DampVelocitySolver.h 
    # ../odesolver/NewmarkImplicitSolver.h

    )
    
set(SOURCE_FILES

    ../initMiscSolver.cpp 
    # ../odesolver/DampVelocitySolver.cpp 
    # ../odesolver/NewmarkImplicitSolver.cpp 
    
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_SOLVER")
AddLinkerDependencies(SofaSimulationTree)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
