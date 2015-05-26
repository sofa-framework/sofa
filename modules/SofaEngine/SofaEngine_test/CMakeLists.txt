include(${SOFA_CMAKE_DIR}/preProject.cmake)


set(HEADER_FILES

    TestEngine.h
        )

set(SOURCE_FILES

    TestEngine.cpp
	Engine_test.cpp
	TransformEngine_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaEngine
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
