cmake_minimum_required(VERSION 2.8)

project("SofaTaucsSolver")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initTaucsSolver.h 
    ../linearsolver/SparseTAUCSSolver.h 
    ../linearsolver/IncompleteTAUCSSolver.h 
           
    )
    
set(SOURCE_FILES

    ../linearsolver/SparseTAUCSSolver.cpp 
    ../linearsolver/IncompleteTAUCSSolver.cpp 
 
    )
    
if(SOFA_EXTLIBS_TAUCS_MT)
	list(APPEND HEADER_FILES "linearsolver/SparseTAUCSLLtSolver.h")

	list(APPEND SOURCE_FILES "linearsolver/SparseTAUCSLLtSolver.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_TAUCS_SOLVER" )
set(LINKER_DEPENDENCIES  )

    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
