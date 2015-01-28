include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	../initEigen2Solver.h 
	# ../linearsolver/EigenMatrixManipulator.h 
	# ../linearsolver/EigenBaseSparseMatrix.h 
 #    ../linearsolver/EigenBaseSparseMatrix_MT.h
	# ../linearsolver/EigenSparseMatrix.h 
	# ../linearsolver/EigenVector.h 
	# ../linearsolver/EigenVectorWrapper.h 
	# ../linearsolver/SVDLinearSolver.h
	)

set(SOURCE_FILES

         ../initEigen2Solver.cpp
	# ../linearsolver/EigenMatrixManipulator.cpp 
	# ../linearsolver/SVDLinearSolver.cpp
	)

include_directories("${SOFA_EXTLIBS_DIR}/newmat")    
set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_EIGEN2_SOLVER")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaDenseSolver)   #eigen-unsupported??

include(${SOFA_CMAKE_DIR}/postProject.cmake)

