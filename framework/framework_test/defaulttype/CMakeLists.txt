cmake_minimum_required(VERSION 2.8)

project(SofaDefaultType_test)

set(SOURCE_FILES
    VecTypesTest.cpp
)

include_directories(${SofaFramework_SOURCE_DIR})

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest_main SofaDefaultType)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${PROJECT_NAME} COMMENT "-> Running unit tests..." VERBATIM)

