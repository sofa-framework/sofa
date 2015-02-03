include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

 #alphabetical order
    AffinePatch_test.cpp
    HexahedronFEMForceField_test.cpp
    LinearElasticity_test.cpp
    TetrahedronFEMForceField_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})

# Define a macro to get the absolute path to a directory;
# absolute paths are perfectly OK for test resources.
AddCompilerDefinitions("SOFASIMPLEFEM_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

AddLinkerDependencies(
    SofaTest
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
