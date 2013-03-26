cmake_minimum_required(VERSION 2.8)

project("SofaDenseSolver")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initDenseSolver.h 
    linearsolver/LULinearSolver.h 
    linearsolver/NewMatVector.h 
    linearsolver/NewMatMatrix.h
    
    )
    
set(SOURCE_FILES

    initDenseSolver.cpp 
    linearsolver/LULinearSolver.cpp 
    linearsolver/NewMatCGLinearSolver.cpp 
    linearsolver/NewMatCholeskySolver.cpp
    
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_DENSE_SOLVER")
set(LINKER_DEPENDENCIES SofaBaseLinearSolver newmat)

include(${SOFA_CMAKE_DIR}/post.cmake)
