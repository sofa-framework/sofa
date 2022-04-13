cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Playback_test)

set(SOURCE_FILES
    ReadState_test.cpp
    WriteState_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Playback)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
