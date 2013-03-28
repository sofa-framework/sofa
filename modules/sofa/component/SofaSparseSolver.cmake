cmake_minimum_required(VERSION 2.8)

project("SofaSparseSolver")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initSparseSolver.h 
    linearsolver/PrecomputedLinearSolver.h 
    linearsolver/PrecomputedLinearSolver.inl 
    linearsolver/SparseCholeskySolver.h 
    linearsolver/SparseLUSolver.h 
    linearsolver/SparseLDLSolverImpl.h 
    linearsolver/SparseLDLSolver.h 
    linearsolver/SparseLDLSolver.inl

    )
    
set(SOURCE_FILES

    initSparseSolver.cpp 
    linearsolver/PrecomputedLinearSolver.cpp 
    linearsolver/SparseCholeskySolver.cpp 
    linearsolver/SparseLUSolver.cpp 
    linearsolver/SparseLDLSolver.cpp
 
    )
    
include_directories("${SOFA_EXTLIBS_DIR}/csparse")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SPARSE_SOLVER" )
set(LINKER_DEPENDENCIES SofaBaseLinearSolver SofaImplicitOdeSolver csparse metis )
    
include(${SOFA_CMAKE_DIR}/post.cmake)
