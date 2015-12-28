cmake_minimum_required(VERSION 2.8.12)

project(SofaConstraint_test)

set(SOURCE_FILES
    BilateralInteractionConstraint_test.cpp
    UncoupledConstraintCorrection_test.cpp)

add_definitions("-DSOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes_test\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
