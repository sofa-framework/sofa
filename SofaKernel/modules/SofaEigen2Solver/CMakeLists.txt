cmake_minimum_required(VERSION 3.12)
project(SofaEigen2Solver LANGUAGES CXX)

set(SOFAEIGEN2SOLVER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAEIGEN2SOLVER_SRC}/config.h.in
    ${SOFAEIGEN2SOLVER_SRC}/initSofaEigen2Solver.h
    ${SOFAEIGEN2SOLVER_SRC}/EigenBaseSparseMatrix.h
    ${SOFAEIGEN2SOLVER_SRC}/EigenMatrixManipulator.h
    ${SOFAEIGEN2SOLVER_SRC}/EigenSparseMatrix.h
    ${SOFAEIGEN2SOLVER_SRC}/EigenVector.h
    ${SOFAEIGEN2SOLVER_SRC}/EigenVectorWrapper.h
    ${SOFAEIGEN2SOLVER_SRC}/SVDLinearSolver.h
)

set(SOURCE_FILES
    ${SOFAEIGEN2SOLVER_SRC}/initSofaEigen2Solver.cpp
    ${SOFAEIGEN2SOLVER_SRC}/EigenMatrixManipulator.cpp
    ${SOFAEIGEN2SOLVER_SRC}/EigenVector.cpp
    ${SOFAEIGEN2SOLVER_SRC}/SVDLinearSolver.cpp
)

sofa_find_package(Eigen3 REQUIRED)
sofa_find_package(SofaBaseLinearSolver REQUIRED)

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

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
