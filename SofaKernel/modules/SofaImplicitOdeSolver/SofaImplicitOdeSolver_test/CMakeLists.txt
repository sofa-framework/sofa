cmake_minimum_required(VERSION 3.1)

project(SofaImplicitOdeSolver_test)

set(SOURCE_FILES
    loadPlugins.cpp
    EulerImplicitSolverStatic_test.cpp
    EulerImplicitSolverDynamic_test.cpp
    SpringSolverDynamic_test.cpp)
    
add_definitions("-DSOFAIMPLICITODESOLVER_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaDeformable SofaValidation)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
