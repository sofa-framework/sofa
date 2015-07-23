include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

 #alphabetical order
    ProjectionToTargetLineMapping_test.cpp
    ProjectionToTargetPlaneMapping_test.cpp
    SubsetMultiMapping_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(
    SofaGTestMain
    SofaTest
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
