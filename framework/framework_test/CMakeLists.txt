cmake_minimum_required(VERSION 2.8.12)

project(SofaFramework_test)

set(SOURCE_FILES
    core/loader/MeshLoader_test.cpp
    core/objectmodel/AspectPool_test.cpp
    core/objectmodel/Data_test.cpp
    defaulttype/MatTypes_test.cpp
    defaulttype/VecTypes_test.cpp
    helper/KdTree_test.cpp
    helper/Utils_test.cpp
    helper/Quater_test.cpp
    helper/system/FileSystem_test.cpp
    helper/system/atomic_test.cpp
    main.cpp
)

include_directories(${gtest_SOURCE_DIR}/include)

add_definitions("-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest_main SofaCore)
#add_dependencies(${PROJECT_NAME} PluginA PluginB PluginC PluginD PluginE PluginF)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${PROJECT_NAME} COMMENT "-> Running unit tests..." VERBATIM)
