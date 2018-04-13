cmake_minimum_required(VERSION 3.1)

project(SofaFramework_test)

set(SOURCE_FILES
    core/loader/MeshLoader_test.cpp
    core/objectmodel/AspectPool_test.cpp
    core/objectmodel/Data_test.cpp
    core/objectmodel/BaseLink_test.cpp
    core/objectmodel/BaseObjectDescription_test.cpp
    core/objectmodel/DataFileName_test.cpp
    core/DataEngine_test.cpp
    defaulttype/MapMapSparseMatrixEigenUtils_test.cpp
    defaulttype/MatTypes_test.cpp
    defaulttype/VecTypes_test.cpp
    helper/types/Color_test.cpp
    helper/types/Material_test.cpp
    helper/KdTree_test.cpp
    helper/Utils_test.cpp
    helper/Quater_test.cpp
    helper/SVector_test.cpp
    helper/vector_test.cpp
    helper/gl/GLSLShader_test.cpp
    helper/io/MeshOBJ_test.cpp
    helper/io/MeshSTL_test.cpp
    helper/system/FileMonitor_test.cpp
    helper/system/FileRepository_test.cpp
    helper/system/FileSystem_test.cpp
    helper/system/PluginManager_test.cpp
    helper/system/atomic_test.cpp
    helper/logging/logging_test.cpp
    testing/TestMessageHandler_test.cpp
    main.cpp
)

#Plugin tests
#add_subdirectory(plugin-system) ## TODO: check if deprecated or... ?
add_subdirectory(plugins)

include_directories(${gtest_SOURCE_DIR}/include)

add_definitions("-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest SofaHelper SofaCore SofaDefaultType)
#add_dependencies(${PROJECT_NAME} PluginA PluginB PluginC PluginD PluginE PluginF)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${PROJECT_NAME} COMMENT "-> Running unit tests..." VERBATIM)
