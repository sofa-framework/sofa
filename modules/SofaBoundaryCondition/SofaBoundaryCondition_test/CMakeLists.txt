include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    AffineMovementConstraint_test.cpp
    PlaneForceField_test.cpp
    ProjectDirectionConstraint_test.cpp
    ProjectToPointConstraint_test.cpp
    FixedConstraint_test.cpp
    PartialFixedConstraint.cpp
    PointConstraint_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaBoundaryCondition
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
