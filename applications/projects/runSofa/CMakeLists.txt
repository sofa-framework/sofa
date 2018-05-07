cmake_minimum_required(VERSION 3.1)
project(runSofa)

option(RUNSOFA_INSTALL_AS_BUNDLE "Create a bundle containing all the installation files (Only Mac OS X for now)" OFF)

set(RESSOURCE_FILES
    resources/docs/runsofa.html
    )

if(APPLE)
    set(RC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/runSOFA.icns")
else()
    set(RC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/sofa.rc")
endif()

include(cmake/GeneratePluginConfig.cmake)
# if MSVC then plugins are located in bin/ instead of lib/ on Mac and Linux
if(MSVC)
	set(_pluginLocation "bin")
else()
	set(_pluginLocation "lib")
endif()
set(_configPluginFileName plugin_list.conf)
set(_defaultConfigPluginFileName "${_configPluginFileName}.default")
set(_defaultConfigPluginFilePath "${CMAKE_BINARY_DIR}/${_pluginLocation}/${_defaultConfigPluginFileName}")
sofa_generate_plugin_config(${_defaultConfigPluginFilePath})
message("Write Plugin list at ${_defaultConfigPluginFilePath}")

if(APPLE)
	set_source_files_properties(${RC_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
endif()

if(APPLE AND RUNSOFA_INSTALL_AS_BUNDLE)
	add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${RC_FILES} Main.cpp )
else()
	add_executable(${PROJECT_NAME} ${RC_FILES} ${RESOURCE_FILES} Main.cpp)
endif()

if(APPLE)
	set_target_properties( ${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_ICON_FILE "runSOFA.icns" )
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC "CONFIG_PLUGIN_FILENAME=${_configPluginFileName}")
target_compile_definitions(${PROJECT_NAME} PUBLIC "DEFAULT_CONFIG_PLUGIN_FILENAME=${_defaultConfigPluginFileName}")
target_link_libraries(${PROJECT_NAME} SofaComponentBase SofaComponentGeneral SofaComponentCommon SofaComponentAdvanced SofaComponentMisc)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(runSofa_test)
    endif()
endif()

if(APPLE AND RUNSOFA_INSTALL_AS_BUNDLE)
	# set(CPACK_COMPONENTS_ALL BundlePack)
	set_property(GLOBAL PROPERTY RUNSOFA_CPACK_COMPONENTS_ALL BundlePack)
	include(cmake/bundle.cmake)
else()
	sofa_install_targets(SofaGui runSofa "")
endif()

# if MSVC then plugins are located in bin/ instead of lib/ on Mac and Linux
file(INSTALL "resources" DESTINATION "../../../share/sofa/gui/runSofa")
install(FILES "${_defaultConfigPluginFilePath}" DESTINATION ${_pluginLocation}/)

