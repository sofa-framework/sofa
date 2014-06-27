include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
	TransformEngine_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaEngine
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
