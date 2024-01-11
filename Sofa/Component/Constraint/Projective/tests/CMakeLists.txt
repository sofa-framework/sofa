cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Constraint.Projective_test)

set(SOURCE_FILES
    AffineMovementProjectiveConstraint_test.cpp
    FixedProjectiveConstraint_test.cpp
    FixedPlaneProjectiveConstraint_test.cpp
    PartialFixedProjectiveConstraint_test.cpp
    DirectionProjectiveConstraint_test.cpp
    LineProjectiveConstraint_test.cpp
    PlaneProjectiveConstraint_test.cpp
    PointProjectiveConstraint_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Topology.Testing SceneCreator)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Constraint.Projective Sofa.Component.StateContainer Sofa.Component.Topology.Container.Dynamic Sofa.Component.Mass Sofa.Component.SolidMechanics.Spring Sofa.Component.MechanicalLoad Sofa.Component.LinearSolver.Iterative Sofa.Component.ODESolver.Backward Sofa.Component.Engine.Select)


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
