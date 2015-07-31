include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
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
    PrecomputedWarpPreconditioner.inl
    )

set(SOURCE_FILES

    initPreconditioner.cpp
    ShewchukPCGLinearSolver.cpp
    JacobiPreconditioner.cpp
    BlockJacobiPreconditioner.cpp
    SSORPreconditioner.cpp
    WarpPreconditioner.cpp
    PrecomputedWarpPreconditioner.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_PRECONDITIONER" )
AddLinkerDependencies(SofaDenseSolver SofaImplicitOdeSolver SofaSimpleFem)

if(SOFA-EXTERNAL_CSPARSE)
    AddLinkerDependencies(SofaSparseSolver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
