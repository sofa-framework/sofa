include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initPreconditioner.h 
    # ../linearsolver/ShewchukPCGLinearSolver.h 
    # ../linearsolver/JacobiPreconditioner.h 
    # ../linearsolver/JacobiPreconditioner.inl 
    # ../linearsolver/BlockJacobiPreconditioner.h 
    # ../linearsolver/BlockJacobiPreconditioner.inl 
    # ../linearsolver/SSORPreconditioner.h 
    # ../linearsolver/SSORPreconditioner.inl 
    # ../linearsolver/WarpPreconditioner.h 
    # ../linearsolver/WarpPreconditioner.inl 
    # ../linearsolver/PrecomputedWarpPreconditioner.h 
    # ../linearsolver/PrecomputedWarpPreconditioner.inl
    )
    
set(SOURCE_FILES

    ../initPreconditioner.cpp 
    # ../linearsolver/ShewchukPCGLinearSolver.cpp 
    # ../linearsolver/JacobiPreconditioner.cpp 
    # ../linearsolver/BlockJacobiPreconditioner.cpp 
    # ../linearsolver/SSORPreconditioner.cpp 
    # ../linearsolver/WarpPreconditioner.cpp 
    # ../linearsolver/PrecomputedWarpPreconditioner.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_PRECONDITIONER" )
AddLinkerDependencies(SofaDenseSolver SofaImplicitOdeSolver SofaSimpleFem )

if(SOFA-EXTERNAL_CSPARSE)
    AddLinkerDependencies(SofaSparseSolver)
endif()
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
