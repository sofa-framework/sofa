include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOFA_TEST_PLUGIN_DIR "${SOFA_BUILD_DIR}/misc/testPlugins")
file(MAKE_DIRECTORY "${SOFA_TEST_PLUGIN_DIR}")

add_subdirectory(plugin-system)

set(SOURCE_FILES
    defaulttype/VecTypesTest.cpp
    helper/system/atomicTest.cpp
    helper/system/FileSystemTest.cpp
    helper/system/UtilsTest.cpp
    core/PluginTest.cpp
    core/objectmodel/AspectPoolTest.cpp
    plugin-system/PluginSystemTest.cpp)

add_definitions("-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources\"")
add_definitions("-DSOFA_TEST_PLUGINS_DIR=\"${SOFA_TEST_PLUGIN_DIR}\"")
add_definitions("-DSOFA_LIB_DIR=\"${SOFA_LIB_DIR}\"")
add_definitions("-DSOFA_BIN_DIR=\"${SOFA_BIN_DIR}\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest_main SofaCore PluginMonitor)
add_dependencies(${PROJECT_NAME} PluginA PluginB PluginC PluginD PluginE PluginF)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
