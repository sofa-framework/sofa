include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../SparsePARDISOSolver.h

    )
set(SOURCE_FILES

    ../SparsePARDISOSolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PARDISO_SOLVER")
set(LINKER_DEPENDENCIES SofaSimulationTree SofaMeshCollision SofaBaseVisual)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
