cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Constraint.Projective_test)

set(SOURCE_FILES
    AffineMovementConstraint_test.cpp
    FixedConstraint_test.cpp
    FixedPlaneConstraint_test.cpp
    PartialFixedConstraint_test.cpp
    ProjectDirectionConstraint_test.cpp
    ProjectToLineConstraint_test.cpp
    ProjectToPlaneConstraint_test.cpp
    ProjectToPointConstraint_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Topology.Testing SceneCreator)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Constraint.Projective Sofa.Component.StateContainer Sofa.Component.Topology.Container.Dynamic Sofa.Component.Mass Sofa.Component.SolidMechanics.Spring Sofa.Component.MechanicalLoad Sofa.Component.LinearSolver.Iterative Sofa.Component.ODESolver.Backward Sofa.Component.Engine.Select)


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
