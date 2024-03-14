cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.MechanicalLoad_test)

set(SOURCE_FILES
    ConstantForceField_test.cpp
    PlaneForceField_test.cpp
    QuadPressureForceField_test.cpp
    SkeletalMotionConstraint_test.cpp
    SurfacePressureForceField_test.cpp
    TrianglePressureForceField_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFA_COMPONENT_MECHANICALLOAD_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.SolidMechanics.Testing SceneCreator)
target_link_libraries(${PROJECT_NAME} Sofa.Component.MechanicalLoad Sofa.Component.StateContainer Sofa.Component.LinearSolver.Iterative Sofa.Component.ODESolver.Backward  Sofa.Component.LinearSolver.Iterative Sofa.Component.Mass Sofa.Component.Topology.Container.Dynamic Sofa.Component.Constraint.Projective)


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
