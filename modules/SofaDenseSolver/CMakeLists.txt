include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initDenseSolver.h
    LULinearSolver.h
    NewMatVector.h
    NewMatMatrix.h

    )

set(SOURCE_FILES

    initDenseSolver.cpp
    LULinearSolver.cpp
    NewMatCGLinearSolver.cpp
    NewMatCholeskySolver.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_DENSE_SOLVER")
AddLinkerDependencies(SofaBaseLinearSolver newmat)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
