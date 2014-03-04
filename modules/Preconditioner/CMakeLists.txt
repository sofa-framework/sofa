include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initPreconditioner.h
    ShewchukPCGLinearSolver.h
    JacobiPreconditioner.h
    JacobiPreconditioner.inl
    BlockJacobiPreconditioner.h
    BlockJacobiPreconditioner.inl
    SSORPreconditioner.h
    SSORPreconditioner.inl
    WarpPreconditioner.h
    WarpPreconditioner.inl
    PrecomputedWarpPreconditioner.h
    PrecomputedWarpPreconditioner.inl)

set(SOURCE_FILES
    initPreconditioner.cpp
    ShewchukPCGLinearSolver.cpp
    JacobiPreconditioner.cpp
    BlockJacobiPreconditioner.cpp
    SSORPreconditioner.cpp
    WarpPreconditioner.cpp
    PrecomputedWarpPreconditioner.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_PRECONDITIONER")
AddLinkerDependencies(DenseSolver ImplicitOdeSolver SimpleFem)

if(SOFA-EXTERNAL_CSPARSE)
    AddLinkerDependencies(SparseSolver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

