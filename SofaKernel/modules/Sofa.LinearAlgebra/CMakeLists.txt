cmake_minimum_required(VERSION 3.12)
project(Sofa.LinearAlgebra LANGUAGES CXX)

set(SOFALINEARALGEBRASRC_ROOT "src/sofa/linearalgebra")

set(HEADER_FILES
    ${SOFALINEARALGEBRASRC_ROOT}/config.h.in
    ${SOFALINEARALGEBRASRC_ROOT}/init.h
    ${SOFALINEARALGEBRASRC_ROOT}/BaseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/BaseVector.h
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/DiagonalMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/FullMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/FullMatrix.inl
    ${SOFALINEARALGEBRASRC_ROOT}/FullVector.h
    ${SOFALINEARALGEBRASRC_ROOT}/FullVector.inl
    ${SOFALINEARALGEBRASRC_ROOT}/matrix_bloc_traits.h
    ${SOFALINEARALGEBRASRC_ROOT}/MatrixExpr.h
    ${SOFALINEARALGEBRASRC_ROOT}/RotationMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/SparseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenBaseSparseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenBaseSparseMatrix_MT.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenMatrixManipulator.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenSparseMatrix.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenVector.h
    ${SOFALINEARALGEBRASRC_ROOT}/EigenVectorWrapper.h

)

set(SOURCE_FILES
    ${SOFALINEARALGEBRASRC_ROOT}/init.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/BaseMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/CompressedRowSparseMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/FullVector.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/BaseVector.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/RotationMatrix.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/EigenMatrixManipulator.cpp
    ${SOFALINEARALGEBRASRC_ROOT}/EigenVector.cpp

)

sofa_find_package(Sofa.Type REQUIRED)
sofa_find_package(Sofa.Helper REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Type Sofa.Helper)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
