cmake_minimum_required(VERSION 3.1)
project(SofaGui)

### Dependencies

## Sofa
find_package(SofaGeneral REQUIRED)
find_package(SofaMisc REQUIRED)

set(SOFAGUI_TARGETS "")

## Version
set(SOFAGUI_VERSION ${SofaFramework_VERSION})

option(SOFAGUI_BUILD_TESTS "Compile the tests that need a GUI (like headlessRecorder_test)." ON)
if(SOFAGUI_BUILD_TESTS AND NOT SOFA_BUILD_TESTS)
    message(WARNING "SOFAGUI_BUILD_TESTS can't be enabled without SOFA_BUILD_TESTS.")
    set(SOFAGUI_BUILD_TESTS OFF CACHE BOOL "Compile the tests that need a GUI (like headlessRecorder_test)." FORCE)
endif()

### Sub-projects

## SofaGuiCommon
add_subdirectory(../applications/sofa/gui/SofaGuiCommon ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiCommon)
list(APPEND SOFAGUI_TARGETS SofaGuiCommon)

## SofaGuiQt
option(SOFAGUI_QT "Compile the Qt GUI" ON)
if(SOFAGUI_QT)
    list(APPEND SOFAGUI_TARGETS SofaGuiQt)
    add_subdirectory(../applications/sofa/gui/qt ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQt)
endif()
sofa_set_01(SOFAGUI_HAVE_SOFAGUIQT VALUE ${SOFAGUI_QT}) # see config.h.in

## OpenGL-based GUIs
if(NOT SOFA_NO_OPENGL)
    ## SofaHeadlessRecorder
    option(SOFAGUI_HEADLESS_RECORDER "Compile the headless recorder (Linux only)" OFF)
    if(SOFAGUI_HEADLESS_RECORDER)
        list(APPEND SOFAGUI_TARGETS SofaHeadlessRecorder)
        add_subdirectory(../applications/sofa/gui/headlessRecorder ${CMAKE_CURRENT_BINARY_DIR}/SofaHeadlessRecorder)
    endif()
    sofa_set_01(SOFAGUI_HAVE_SOFAHEADLESSRECORDER VALUE ${SOFAGUI_HEADLESS_RECORDER}) # see config.h.in
endif()

## SofaGuiMain
list(APPEND SOFAGUI_TARGETS SofaGuiMain)
add_subdirectory(../applications/sofa/gui/SofaGuiMain ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiMain)

## IDE folder
foreach(TARGET ${SOFAGUI_TARGETS})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaGui")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../applications>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
    # target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${QT_INCLUDES}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${TARGET} PROPERTIES VERSION ${SOFAGUI_VERSION})
endforeach()

## SofaGui/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.h" DESTINATION "include/${PROJECT_NAME}/${PROJECT_NAME}" COMPONENT headers)

## CMake package configurations files
sofa_write_package_config_files(SofaGui ${SOFAGUI_VERSION})
