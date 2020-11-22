cmake_minimum_required(VERSION 3.12)
project(SofaBaseLinearSolver)

set(HEADER_FILES
    BlocMatrixWriter.h
    CGLinearSolver.h
    CGLinearSolver.inl
    CompressedRowSparseMatrix.h
    CompressedRowSparseMatrix.inl
    DefaultMultiMatrixAccessor.h
    DiagonalMatrix.h
    FullMatrix.h
    FullVector.h
    GraphScatteredTypes.h
    MatrixExpr.h
    MatrixLinearSolver.h
    MatrixLinearSolver.inl
    SingleMatrixAccessor.h
    SparseMatrix.h
    config.h
    initBaseLinearSolver.h
    matrix_bloc_traits.h
)

set(SOURCE_FILES
    CGLinearSolver.cpp
    DefaultMultiMatrixAccessor.cpp
    FullVector.cpp
    GraphScatteredTypes.cpp
    MatrixLinearSolver.cpp
    SingleMatrixAccessor.cpp
    initBaseLinearSolver.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaBase
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
