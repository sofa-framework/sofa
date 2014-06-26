cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "SofaFramework_test")
project(${PROJECT_NAME})

set(SOURCE_FILES
    defaulttype/VecTypesTest.cpp
    helper/system/atomicTest.cpp
    helper/system/FileSystemTest.cpp
    helper/system/UtilsTest.cpp
    core/PluginTest.cpp
    core/objectmodel/AspectPoolTest.cpp
)

include_directories(${gtest_SOURCE_DIR}/include)

add_definitions("-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_test(${PROJECT_NAME} COMMAND ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} gtest_main SofaCore)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${PROJECT_NAME} COMMENT "-> Running unit tests..." VERBATIM)

