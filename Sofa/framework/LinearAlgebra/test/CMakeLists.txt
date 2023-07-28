cmake_minimum_required(VERSION 3.12)

project(Sofa.LinearAlgebra_test)

set(SOURCE_FILES
    BTDMatrix_test.cpp
    BaseMatrix_test.cpp
    CompressedRowSparseMatrix_test.cpp
    Matrix_test.cpp
    RotationMatrix_test.cpp
    SparseMatrixProduct_test.cpp
    SparseMatrixStorageOrder_test.cpp
    TriangularSystemSolver_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.LinearAlgebra Sofa.LinearAlgebra.Testing Sofa.DefaultType)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
