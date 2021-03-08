cmake_minimum_required(VERSION 3.12)

project(SofaMiscSolver_test)

sofa_find_package(SofaMiscSolver REQUIRED)

set(SOURCE_FILES
    NewmarkImplicitSolverDynamic_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaMiscSolver)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
