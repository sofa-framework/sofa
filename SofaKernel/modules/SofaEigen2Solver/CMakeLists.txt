cmake_minimum_required(VERSION 3.12)
project(SofaEigen2Solver)

set(HEADER_FILES
    EigenBaseSparseMatrix.h
    EigenMatrixManipulator.h
    EigenSparseMatrix.h
    EigenVector.h
    EigenVectorWrapper.h
    SVDLinearSolver.h
    config.h.in
    initEigen2Solver.h
)

set(SOURCE_FILES
    EigenMatrixManipulator.cpp
    EigenVector.cpp
    SVDLinearSolver.cpp
    initEigen2Solver.cpp
)

find_package(Eigen3 REQUIRED)

if (SOFA_OPENMP AND "${EIGEN3_VERSION}" VERSION_LESS 3.2.9)
    sofa_find_package(OpenMP BOTH_SCOPES) # will set/update SOFAEIGEN2SOLVER_HAVE_OPENMP
endif()

if (SOFAEIGEN2SOLVER_HAVE_OPENMP)
    list(APPEND HEADER_FILES EigenBaseSparseMatrix_MT.h)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

if (SOFAEIGEN2SOLVER_HAVE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

sofa_add_targets_to_package(
    PACKAGE_NAME SofaBase
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
