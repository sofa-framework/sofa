cmake_minimum_required(VERSION 2.8)

project(SofaExplicitOdeSolver_test)

set(SOURCE_FILES
    CentralDifferenceExplicitSolverDynamic_test.cpp
    EulerExplicitSolverDynamic_test.cpp
    RungeKutta2ExplicitSolverDynamic_test.cpp
    RungeKutta4ExplicitSolverDynamic_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
