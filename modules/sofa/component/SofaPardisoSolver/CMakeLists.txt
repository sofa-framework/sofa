include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../linearsolver/SparsePARDISOSolver.h

    )
set(SOURCE_FILES

    ../linearsolver/SparsePARDISOSolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_PARDISO_SOLVER")
AddLinkerDependencies(SofaSimulationTree SofaMeshCollision SofaBaseVisual)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
