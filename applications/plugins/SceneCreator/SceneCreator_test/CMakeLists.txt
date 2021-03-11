cmake_minimum_required(VERSION 3.12)

project(SceneCreator_test)

sofa_find_package(SofaMiscFem REQUIRED)

set(HEADER_FILES

)

set(SOURCE_FILES
    SceneCreator_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SceneCreator SofaGTestMain SofaTest SofaMiscFem)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
