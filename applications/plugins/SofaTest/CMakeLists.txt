include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initTestPlugin.h
    Sofa_test.h
    Solver_test.h
    Mapping_test.h
    ProjectiveConstraintSet_test.h

	)

set(SOURCE_FILES

    initTestPlugin.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddLinkerDependencies(
    SceneCreator
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
