cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initSparseSolver.h 
    ../linearsolver/PrecomputedLinearSolver.h 
    ../linearsolver/PrecomputedLinearSolver.inl 
    ../linearsolver/SparseCholeskySolver.h 
    ../linearsolver/SparseLUSolver.h 
    ../linearsolver/SparseLDLSolverImpl.h 
    ../linearsolver/SparseLDLSolver.h 
    ../linearsolver/SparseLDLSolver.inl
    )
    
set(SOURCE_FILES

    ../initSparseSolver.cpp 
    ../linearsolver/PrecomputedLinearSolver.cpp 
    ../linearsolver/SparseCholeskySolver.cpp 
    ../linearsolver/SparseLUSolver.cpp 
    ../linearsolver/SparseLDLSolver.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SPARSE_SOLVER" )
set(LINKER_DEPENDENCIES SofaBaseLinearSolver SofaImplicitOdeSolver csparse)

if(SOFA-EXTERNAL_HAVE_METIS)
	list(APPEND LINKER_DEPENDENCIES metis)
endif()
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
