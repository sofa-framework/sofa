cmake_minimum_required(VERSION 3.12)

project(SofaBaseLinearSolver_test)

set(SOURCE_FILES
    Matrix_test.cpp
    BaseMatrix_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaBaseLinearSolver SofaEigen2Solver SofaDefaultType.Testing)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
