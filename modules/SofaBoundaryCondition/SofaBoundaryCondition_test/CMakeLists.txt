include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    PlaneForceField_test.cpp
    AffineMovementConstraint_test.cpp
    ProjectToPointConstraint_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaBoundaryCondition
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
