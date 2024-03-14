cmake_minimum_required(VERSION 3.12)
project(Sofa.LinearAlgebra LANGUAGES CXX)

set(SOFALINEARALGEBRASRC_ROOT "src/sofa/linearalgebra")

set(HEADER_FILES
    ${SOFALINEARALGEBRASRC_ROOT}/config.h.in
    ${SOFALINEARALGEBRASRC_ROOT}/fwd.h
    ${SOFALINEARALGEBRASRC_ROOT}/init.h
    ${SOFALINEARALGEBRASRC_ROOT}/BTDMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/BTDMatrix.inl
    ${SOFALINEARALGEBRASRC_ROOT}/BaseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/BaseVector.h
    ${SOFALINEARALGEBRASRC_ROOT}/BlockFullMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/BlockFullMatrix.inl
    ${SOFALINEARALGEBRASRC_ROOT}/BlocFullMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/BlocFullMatrix.inl
    ${SOFALINEARALGEBRASRC_ROOT}/BlockDiagonalMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/BlockDiagonalMatrix.inl
    ${SOFALINEARALGEBRASRC_ROOT}/BlockVector.h
    ${SOFALINEARALGEBRASRC_ROOT}/BlockVector.inl
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrixConstraint.h
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrixConstraintEigenUtils.h
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrixGeneric.h
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrixMechanical.h
    ${SOFALINEARALGEBRASRC_ROOT}/DiagonalMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/DiagonalSystemSolver.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenBaseSparseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenSparseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenVector.h
    ${SOFALINEARALGEBRASRC_ROOT}/FullMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/FullMatrix.inl
    ${SOFALINEARALGEBRASRC_ROOT}/FullVector.h
    ${SOFALINEARALGEBRASRC_ROOT}/FullVector.inl
    ${SOFALINEARALGEBRASRC_ROOT}/MatrixExpr.h
    ${SOFALINEARALGEBRASRC_ROOT}/RotationMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrixProduct.h
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrixProduct[CompressedRowSparseMatrix].h
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrixProduct[EigenSparseMatrix].h
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrixStorageOrder.h
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrixStorageOrder[EigenSparseMatrix].h
    ${SOFALINEARALGEBRASRC_ROOT}/TriangularSystemSolver.h
    ${SOFALINEARALGEBRASRC_ROOT}/matrix_bloc_traits.h
)

set(SOURCE_FILES
    ${SOFALINEARALGEBRASRC_ROOT}/BaseMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/BaseVector.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/BlockDiagonalMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/BlockFullMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/BlockVector.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/BTDMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrixConstraint.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrixGeneric.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrixMechanical.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/EigenVector.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/FullMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/FullVector.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/RotationMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrixProduct[CompressedRowSparseMatrix].cpp
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrixProduct[EigenSparseMatrix].cpp
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrixStorageOrder[EigenSparseMatrix].cpp
    ${SOFALINEARALGEBRASRC_ROOT}/init.cpp
)

sofa_find_package(Eigen3 REQUIRED)

if (SOFA_OPENMP AND "${EIGEN3_VERSION}" VERSION_LESS 3.2.10)
    sofa_find_package(OpenMP BOTH_SCOPES) # will set/update SOFA_LINEARALGEBRA_HAVE_OPENMP
endif()

if (SOFA_LINEARALGEBRA_HAVE_OPENMP)
    list(APPEND HEADER_FILES ${SOFALINEARALGEBRASRC_ROOT}/EigenBaseSparseMatrix_MT.h)
endif()

sofa_find_package(Sofa.Type REQUIRED)
sofa_find_package(Sofa.Helper REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Type Sofa.Helper Eigen3::Eigen)

if (SOFA_LINEARALGEBRA_HAVE_OPENMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_LINEARALGEBRA_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_LINEARALGEBRA_BUILD_TESTS)
    add_subdirectory(Testing)
    add_subdirectory(test)
endif()
