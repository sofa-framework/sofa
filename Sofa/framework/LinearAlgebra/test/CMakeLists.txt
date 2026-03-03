cmake_minimum_required(VERSION 3.22)

project(Sofa.LinearAlgebra_test)

set(SOURCE_FILES
    BlockFullMatrix_test.cpp
    BTDMatrix_test.cpp
    BaseMatrix_test.cpp
    CompressedRowSparseMatrix_test.cpp
    CompressedRowSparseMatrixGeneric_test.cpp
    CompressedRowSparseMatrixConstraint_test.cpp
    CompressedRowSparseMatrixMechanical_test.cpp
    Matrix_test.cpp
    RotationMatrix_test.cpp
    SparseMatrixProduct_test.cpp
    SparseMatrixStorageOrder_test.cpp
    TriangularSystemSolver_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.LinearAlgebra Sofa.LinearAlgebra.Testing Sofa.DefaultType)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
