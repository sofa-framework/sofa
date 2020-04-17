cmake_minimum_required(VERSION 3.1)
project(runSofa)

if(APPLE)
    set(RC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/runSofa.icns")
else()
    set(RC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/runSofa.rc")
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

configure_file(${_defaultConfigPluginFilePath} ${PROJECT_BINARY_DIR}/${_defaultConfigPluginFileName} COPYONLY)

set(RESOURCE_FILES
    resources/docs/runsofa.html
    ${PROJECT_BINARY_DIR}/${_defaultConfigPluginFileName}
    )

if(APPLE)
    set_source_files_properties(${RC_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
endif()

set(HEADER_FILES
    runSofa_config.h.in
    runSofaValidation.h
)

set(SOURCE_FILES
    Main.cpp
)

find_package(SofaValidation QUIET)

if(SofaValidation_FOUND)
    list(APPEND SOURCE_FILES runSofaValidation.cpp)
else()
    list(APPEND SOURCE_FILES runSofaNoValidation.cpp)
endif()

if(APPLE AND SOFA_BUILD_APP_BUNDLE)
    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${RC_FILES} ${RESOURCE_FILES} ${HEADER_FILES} ${SOURCE_FILES})
    set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/runSofa.plist")
else()
    add_executable(${PROJECT_NAME} ${RC_FILES} ${RESOURCE_FILES} ${HEADER_FILES} ${SOURCE_FILES})
endif()

target_compile_definitions(${PROJECT_NAME} PUBLIC "CONFIG_PLUGIN_FILENAME=${_configPluginFileName}")
target_compile_definitions(${PROJECT_NAME} PUBLIC "DEFAULT_CONFIG_PLUGIN_FILENAME=${_defaultConfigPluginFileName}")
target_link_libraries(${PROJECT_NAME} SofaAdvanced SofaMisc)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)

if(SofaValidation_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaValidation)
else()
    message(WARNING "runSofa: SofaValidation has not been found ; disabling validation feature.")
endif()

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(runSofa_test)
    endif()
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/>")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/runSofa_config.h.in" "${CMAKE_BINARY_DIR}/include/runSofa_config.h")

# Create build and install versions of .ini file for resources finding
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in" "${CMAKE_BINARY_DIR}/etc/${PROJECT_NAME}.ini")
set(RESOURCES_DIR "../share/sofa/gui/runSofa")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in" "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini" DESTINATION etc RENAME ${PROJECT_NAME}.ini COMPONENT applications)

install(DIRECTORY "resources/" DESTINATION "share/sofa/gui/runSofa" COMPONENT resources)
install(FILES "${_defaultConfigPluginFilePath}" DESTINATION ${_pluginLocation}/ COMPONENT applications)

sofa_install_targets(SofaGui runSofa runSofa)

# Post-install scripts
if(SOFA_BUILD_RELEASE_PACKAGE)
    if(CMAKE_SYSTEM_NAME STREQUAL Linux)
        install(CODE "file(WRITE
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/qt.conf\"
            \"[Paths]\\n  Prefix = ..\\n  LibraryExecutables = bin\"
            )"
            COMPONENT applications)
        install(CODE "execute_process(COMMAND /bin/bash
            \"${CMAKE_SOURCE_DIR}/scripts/linux-postinstall-fixup.sh\"
            \"${CMAKE_BINARY_DIR}\"
            \"\$ENV{DESTDIR\}\${CMAKE_INSTALL_PREFIX\}\"
            \"${Qt5_DIR}/../../..\"
            )"
            COMPONENT applications)
    elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
        if(SOFA_BUILD_APP_BUNDLE)
            include(macdeployqt) # set MACDEPLOYQT_EXECUTABLE
            install(CODE "execute_process(COMMAND /bin/bash
                \"${CMAKE_SOURCE_DIR}/scripts/macos-postinstall-fixup.sh\"
                \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/../..\"
                \"${Qt5_DIR}/../../..\"
                \"${MACDEPLOYQT_EXECUTABLE}\"
                )"
                COMPONENT applications)
        else()
            install(CODE "execute_process(COMMAND /bin/bash
                \"${CMAKE_SOURCE_DIR}/scripts/macos-postinstall-fixup.sh\"
                \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\"
                \"${Qt5_DIR}/../../..\"
                )"
                COMPONENT applications)
        endif()
    endif()
endif()
