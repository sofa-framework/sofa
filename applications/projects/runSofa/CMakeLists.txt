cmake_minimum_required(VERSION 3.12)
project(runSofa)

###################################
# Generate plugin_list.conf.default
include(cmake/GeneratePluginConfig.cmake)
if(MSVC)
    # plugins are located in bin/
    set(_pluginLocation "bin")
else()
    # plugins are located in lib/
    set(_pluginLocation "lib")
endif()
set(_configPluginFileName plugin_list.conf)
set(_defaultConfigPluginFileName "${_configPluginFileName}.default")
set(_defaultConfigPluginFilePath "${CMAKE_BINARY_DIR}/${_pluginLocation}/${_defaultConfigPluginFileName}")
sofa_generate_plugin_config(${_defaultConfigPluginFilePath})
message("Write Plugin list at ${_defaultConfigPluginFilePath}")
configure_file(${_defaultConfigPluginFilePath} ${PROJECT_BINARY_DIR}/${_defaultConfigPluginFileName} COPYONLY)
install(FILES "${_defaultConfigPluginFilePath}" DESTINATION ${_pluginLocation}/ COMPONENT applications)
###################################

sofa_find_package(Sofa.Component.Playback QUIET)

sofa_find_package(Sofa.Simulation.Graph REQUIRED)
sofa_find_package(Sofa.GUI.Common REQUIRED)
sofa_find_package(Sofa.GUI.Batch REQUIRED)
sofa_find_package(SceneChecking REQUIRED)

set(HEADER_FILES
    runSofaValidation.h
    )
set(SOURCE_FILES
    Main.cpp
    )

if(Sofa.Component.Playback_FOUND)
    list(APPEND SOURCE_FILES runSofaValidation.cpp)
else()
    list(APPEND SOURCE_FILES runSofaNoValidation.cpp)
endif()

set(RESOURCE_FILES
    resources/docs/runsofa.html
    ${PROJECT_BINARY_DIR}/${_defaultConfigPluginFileName}
    )
if(APPLE)
    set(RC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/runSofa.icns")
    set_source_files_properties(${RC_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
else()
    set(RC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/runSofa.rc")
endif()

if(APPLE AND SOFA_BUILD_APP_BUNDLE)
    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${RC_FILES} ${RESOURCE_FILES} ${HEADER_FILES} ${SOURCE_FILES})
    set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/runSofa.plist")
else()
    add_executable(${PROJECT_NAME} ${RC_FILES} ${RESOURCE_FILES} ${HEADER_FILES} ${SOURCE_FILES})
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC "CONFIG_PLUGIN_FILENAME=${_configPluginFileName}")
target_compile_definitions(${PROJECT_NAME} PUBLIC "DEFAULT_CONFIG_PLUGIN_FILENAME=${_defaultConfigPluginFileName}")

target_link_libraries(${PROJECT_NAME} Sofa.Simulation.Graph)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Common)
target_link_libraries(${PROJECT_NAME} Sofa.GUI.Batch)
target_link_libraries(${PROJECT_NAME} SceneChecking)

if(Sofa.Component.Playback_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.Component.Playback)
else()
    message(WARNING "runSofa: Sofa.Component.Playback not found, disabling validation feature.")
endif()

if(SOFA_BUILD_TESTS)
    add_subdirectory(runSofa_test)
endif()

# Create build and install versions of .ini file for resources finding
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in" "${CMAKE_BINARY_DIR}/etc/${PROJECT_NAME}.ini")
set(RESOURCES_DIR "../share/sofa/gui/runSofa")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in" "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini" DESTINATION etc RENAME ${PROJECT_NAME}.ini COMPONENT applications)

install(DIRECTORY "resources/" DESTINATION "share/sofa/gui/runSofa" COMPONENT resources)

sofa_add_targets_to_package(
    PACKAGE_NAME Sofa.GUI
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    INCLUDE_INSTALL_DIR "Sofa.GUI/runSofa"
    )
