cmake_minimum_required(VERSION 3.1)

project(SofaBoundaryCondition_test)

set(SOURCE_FILES ../../empty.cpp)

list(APPEND SOURCE_FILES
    AffineMovementConstraint_test.cpp
    FixedConstraint_test.cpp
    PartialFixedConstraint.cpp
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
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SceneCreator)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
