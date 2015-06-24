include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    Node_test.h
    tree/GNode_test.cpp
    graph/DAG_test.cpp
    graph/Node_test.cpp
    graph/Simulation_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Define a macro to get the absolute path to a directory;
# absolute paths are perfectly OK for test resources.
AddCompilerDefinitions("SOFASIMULATION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")


AddLinkerDependencies(
    SofaTest
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
