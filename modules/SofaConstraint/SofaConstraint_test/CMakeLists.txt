include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    BilateralInteractionConstraint_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaConstraint
)

AddCompilerDefinitions("SOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes_test\"")
include(${SOFA_CMAKE_DIR}/postProject.cmake)

