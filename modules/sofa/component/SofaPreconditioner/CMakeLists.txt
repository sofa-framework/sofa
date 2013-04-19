cmake_minimum_required(VERSION 2.8)

project("SofaPreconditioner")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initPreconditioner.h 
    ../linearsolver/ShewchukPCGLinearSolver.h 
    ../linearsolver/JacobiPreconditioner.h 
    ../linearsolver/JacobiPreconditioner.inl 
    ../linearsolver/BlockJacobiPreconditioner.h 
    ../linearsolver/BlockJacobiPreconditioner.inl 
    ../linearsolver/SSORPreconditioner.h 
    ../linearsolver/SSORPreconditioner.inl 
    ../linearsolver/WarpPreconditioner.h 
    ../linearsolver/WarpPreconditioner.inl 
    ../linearsolver/PrecomputedWarpPreconditioner.h 
    ../linearsolver/PrecomputedWarpPreconditioner.inl
    )
    
set(SOURCE_FILES

    ../initPreconditioner.cpp 
    ../linearsolver/ShewchukPCGLinearSolver.cpp 
    ../linearsolver/JacobiPreconditioner.cpp 
    ../linearsolver/BlockJacobiPreconditioner.cpp 
    ../linearsolver/SSORPreconditioner.cpp 
    ../linearsolver/WarpPreconditioner.cpp 
    ../linearsolver/PrecomputedWarpPreconditioner.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PRECONDITIONER" )
set(LINKER_DEPENDENCIES SofaDenseSolver SofaImplicitOdeSolver SofaSimpleFem )

if(EXTERNAL_HAVE_CSPARSE)
    include_directories("${SOFA_EXTLIBS_DIR}/csparse")
    list(APPEND LINKER_DEPENDENCIES SofaSparseSolver)
endif()
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
