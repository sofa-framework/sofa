cmake_minimum_required(VERSION 2.8.12)
project(SofaEigen2Solver)

set(HEADER_FILES
    EigenBaseSparseMatrix.h
    EigenBaseSparseMatrix_MT.h
    EigenMatrixManipulator.h
    EigenSparseMatrix.h
    # EigenSparseSquareMatrix.h
    EigenVector.h
    EigenVectorWrapper.h
    SVDLinearSolver.h
    config.h
    initEigen2Solver.h
)

set(SOURCE_FILES
    EigenMatrixManipulator.cpp
    SVDLinearSolver.cpp
    initEigen2Solver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_EIGEN2_SOLVER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
