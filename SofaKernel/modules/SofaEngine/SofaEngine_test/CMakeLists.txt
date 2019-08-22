cmake_minimum_required(VERSION 3.1)

project(SofaEngine_test)


set(HEADER_FILES
    TestEngine.h
        )

set(SOURCE_FILES
    Engine_test.cpp
    TestEngine.cpp
    BoxROI_test.cpp
    )


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
