include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	initEigen2Solver.h
	EigenMatrixManipulator.h
	EigenBaseSparseMatrix.h
        EigenBaseSparseMatrix_MT.h
	EigenSparseMatrix.h
#    EigenSparseSquareMatrix.h
	EigenVector.h
	EigenVectorWrapper.h
	SVDLinearSolver.h
	)

set(SOURCE_FILES

	initEigen2Solver.cpp
	EigenMatrixManipulator.cpp
	SVDLinearSolver.cpp
	)

include_directories("${SOFA_EXTLIBS_DIR}/newmat")
set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_EIGEN2_SOLVER")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaDenseSolver)   #eigen-unsupported??

include(${SOFA_CMAKE_DIR}/postProject.cmake)

