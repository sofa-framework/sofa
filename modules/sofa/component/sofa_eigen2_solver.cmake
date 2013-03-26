cmake_minimum_required(VERSION 2.8)

project("SofaEigen2Solver")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

	initEigen2Solver.h 
	linearsolver/EigenMatrixManipulator.h 
	linearsolver/EigenBaseSparseMatrix.h 
	linearsolver/EigenSparseMatrix.h 
#    linearsolver/EigenSparseSquareMatrix.h 
	linearsolver/EigenVector.h 
	linearsolver/EigenVectorWrapper.h 
	linearsolver/SVDLinearSolver.h
	)

set(SOURCE_FILES

	initEigen2Solver.cpp 
	linearsolver/EigenMatrixManipulator.cpp 
	linearsolver/SVDLinearSolver.cpp
	)

include_directories("${SOFA_EXTLIBS_DIR}/newmat")    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILE_DEFINES "SOFA_BUILD_EIGEN2_SOLVER")
set(LINK_DEPENDENCIES SofaDenseSolver)   #eigen-unsupported??

include(${SOFA_CMAKE_DIR}/post.cmake)

