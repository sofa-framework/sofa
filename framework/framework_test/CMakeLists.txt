cmake_minimum_required(VERSION 2.8)

project(SofaFramework_test)

set(SOURCE_FILES
    main.cpp
    defaulttype/VecTypesTest.cpp
    #defaulttype/MatTypesTest.cpp
    helper/system/atomicTest.cpp
    helper/system/FileSystemTest.cpp
    helper/system/UtilsTest.cpp
    #core/PluginTest.cpp
    core/objectmodel/AspectPoolTest.cpp
    #plugin-system/PluginSystemTest.cpp
    )

include_directories(${gtest_SOURCE_DIR}/include)

add_definitions("-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest_main SofaCore)
#add_dependencies(${PROJECT_NAME} PluginA PluginB PluginC PluginD PluginE PluginF)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${PROJECT_NAME} COMMENT "-> Running unit tests..." VERBATIM)

#add_subdirectory(defaulttype)
