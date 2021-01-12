cmake_minimum_required(VERSION 3.12)

project(SofaExplicitOdeSolver_test)

set(SOURCE_FILES
    EulerExplicitSolverDynamic_test.cpp)

find_package(SofaExplicitOdeSolver REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaExplicitOdeSolver)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
