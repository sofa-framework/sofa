cmake_minimum_required(VERSION 3.1)

project(SofaGeneralExplicitOdeSolver_test)

set(SOURCE_FILES ../../empty.cpp)

list(APPEND SOURCE_FILES
    CentralDifferenceExplicitSolverDynamic_test.cpp
    RungeKutta2ExplicitSolverDynamic_test.cpp
    RungeKutta4ExplicitSolverDynamic_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
