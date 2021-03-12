cmake_minimum_required(VERSION 3.12)

project(SofaConstraint_test)

sofa_find_package(SofaConstraint REQUIRED)

set(SOURCE_FILES ../../empty.cpp)

list(APPEND SOURCE_FILES
    #LocalMinDistance_test.cpp
    GenericConstraintSolver_test.cpp
    BilateralInteractionConstraint_test.cpp
    UncoupledConstraintCorrection_test.cpp)

add_definitions("-DSOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes_test\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaConstraint)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
