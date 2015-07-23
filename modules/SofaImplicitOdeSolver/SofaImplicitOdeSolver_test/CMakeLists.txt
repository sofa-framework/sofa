cmake_minimum_required(VERSION 2.8)

project(SofaImplicitOdeSolver_test)

set(SOURCE_FILES
    EulerImplicitSolverStatic_test.cpp
    EulerImplicitSolverDynamic_test.cpp
    SpringSolverDynamic_test.cpp
    VariationalSymplecticExplicitSolverDynamic_test.cpp
    VariationalSymplecticImplicitSolverDynamic_test.cpp
    VariationalSymplecticImplicitSolverNonLinearForceDynamic_test.cpp)

add_definitions("-DSOFAIMPLICITODESOLVER_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
