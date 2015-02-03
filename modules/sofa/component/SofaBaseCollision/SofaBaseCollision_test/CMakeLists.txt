include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    BroadPhase_test.h
)

set(SOURCE_FILES

 #alphabetical order
    BroadPhase_test.cpp
    OBB_test.cpp
    Sphere_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(
    SofaTest
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
