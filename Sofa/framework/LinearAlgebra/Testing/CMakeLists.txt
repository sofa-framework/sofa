cmake_minimum_required(VERSION 3.22)
project(Sofa.LinearAlgebra.Testing LANGUAGES CXX)

set(HEADER_FILES
    src/Sofa.LinearAlgebra.Testing/BaseMatrix_test.h
    src/Sofa.LinearAlgebra.Testing/SparseMatrixTest.h
)

add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME} INTERFACE Sofa.Testing)
target_include_directories(${PROJECT_NAME} INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")
