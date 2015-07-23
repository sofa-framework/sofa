include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

 #alphabetical order
    TopologicalChangeProcessor_test.cpp
)

# Define a macro to get the absolute path to a directory;
# absolute paths are perfectly OK for test resources.
AddCompilerDefinitions("SOFAMISCTOPOLOGY_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(
    SofaGTestMain
    SofaTest
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
