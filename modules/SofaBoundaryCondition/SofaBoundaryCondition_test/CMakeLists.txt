cmake_minimum_required(VERSION 3.1)

project(SofaBoundaryCondition_test)

sofa_find_package(SofaSparseSolver QUIET)

set(HEADER_FILES
    config.h.in
    )
set(SOURCE_FILES
    AffineMovementConstraint_test.cpp
    FixedConstraint_test.cpp
    PartialFixedConstraint_test.cpp
    PlaneForceField_test.cpp
    PointConstraint_test.cpp
    ProjectDirectionConstraint_test.cpp
    ProjectToLineConstraint_test.cpp
    ProjectToPlaneConstraint_test.cpp
    ProjectToPointConstraint_test.cpp
    QuadPressureForceField_test.cpp
    SkeletalMotionConstraint_test.cpp
    TrianglePressureForceField_test.cpp
    ConstantForceField_test.cpp
    loadPlugins.cpp
    )

add_definitions("-DSOFABOUNDARYCONDITION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SceneCreator)
if(SofaSparseSolver_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaSparseSolver)
endif()

configure_file(config.h.in "${PROJECT_BINARY_DIR}/include/SofaBoundaryCondition_test/config.h")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
