cmake_minimum_required(VERSION 2.8)

project(SofaEngine_test)


set(HEADER_FILES

    TestEngine.h
        )

set(SOURCE_FILES

    TestEngine.cpp
	Engine_test.cpp
	TransformEngine_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest_main SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
