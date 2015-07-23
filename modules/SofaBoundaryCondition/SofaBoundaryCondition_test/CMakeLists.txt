include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
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
        )

# Define a macro to get the absolute path to a directory;
# absolute paths are perfectly OK for test resources.
AddCompilerDefinitions("SOFABOUNDARYCONDITION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaGTestMain
    SofaTest
    SofaBoundaryCondition
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
