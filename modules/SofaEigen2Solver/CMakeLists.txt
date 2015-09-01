cmake_minimum_required(VERSION 2.8.11)
project(SofaEigen2Solver)

set(HEADER_FILES

    config.h
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


set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} PUBLIC SofaBaseLinearSolver)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/SofaGeneral>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAEIGEN2SOLVER_COMPILER_FLAGS "-DSOFA_BUILD_EIGEN2_SOLVER")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAEIGEN2SOLVER_COMPILER_FLAGS}")
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaGeneral_libraries
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}"

)
