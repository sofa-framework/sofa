include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initSparseSolver.h
    PrecomputedLinearSolver.h
    PrecomputedLinearSolver.inl
    SparseCholeskySolver.h
    SparseLUSolver.h
    SparseLDLSolverImpl.h
    SparseLDLSolver.h
    SparseLDLSolver.inl
    )

set(SOURCE_FILES

    initSparseSolver.cpp
    PrecomputedLinearSolver.cpp
    SparseCholeskySolver.cpp
    SparseLUSolver.cpp
    SparseLDLSolver.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SPARSE_SOLVER" )
AddLinkerDependencies(SofaBaseLinearSolver SofaImplicitOdeSolver csparse)

if(SOFA-EXTERNAL_METIS)
        AddLinkerDependencies(metis)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
