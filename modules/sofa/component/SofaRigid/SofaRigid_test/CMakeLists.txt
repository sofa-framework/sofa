include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
 #alphabetical order
    RigidMapping_test.cpp
    RigidRigidMapping_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(
    SofaTest
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
