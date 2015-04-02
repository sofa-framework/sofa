include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    SparsePARDISOSolver.h

    )
set(SOURCE_FILES

    SparsePARDISOSolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_PARDISO_SOLVER")
AddLinkerDependencies(SofaSimulationTree SofaMeshCollision SofaBaseVisual)
AddLinkerDependencies(pardiso500-GNU461-X86-64 gfortran blas lapack gomp)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
