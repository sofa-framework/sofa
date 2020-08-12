cmake_minimum_required(VERSION 3.12)
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
add_subdirectory(../modules/SofaGuiCommon ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiCommon)
list(APPEND SOFAGUI_TARGETS SofaGuiCommon)

## SofaGuiQt
option(SOFAGUI_QT "Compile the Qt GUI" ON)
if(SOFAGUI_QT)
    list(APPEND SOFAGUI_TARGETS SofaGuiQt)
    add_subdirectory(../modules/SofaGuiQt ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQt)
endif()
sofa_set_01(SOFAGUI_HAVE_SOFAGUIQT VALUE ${SOFAGUI_QT}) # see config.h.in

## OpenGL-based GUIs
if(NOT SOFA_NO_OPENGL)
    ## SofaHeadlessRecorder
    option(SOFAGUI_HEADLESS_RECORDER "Compile the headless recorder (Linux only)" OFF)
    if(SOFAGUI_HEADLESS_RECORDER)
        list(APPEND SOFAGUI_TARGETS SofaHeadlessRecorder)
        add_subdirectory(../modules/SofaHeadlessRecorder ${CMAKE_CURRENT_BINARY_DIR}/SofaHeadlessRecorder)
    endif()
    sofa_set_01(SOFAGUI_HAVE_SOFAHEADLESSRECORDER VALUE ${SOFAGUI_HEADLESS_RECORDER}) # see config.h.in
endif()

## SofaGuiMain
list(APPEND SOFAGUI_TARGETS SofaGuiMain)
add_subdirectory(../modules/SofaGuiMain ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiMain)

set(SOFAGUI_SRC src/${PROJECT_NAME})
set(HEADER_FILES
    ${SOFAGUI_SRC}/config.h.in
    ${SOFAGUI_SRC}/initSofaGui.h
    )
set(SOURCE_FILES
    ${SOFAGUI_SRC}/initSofaGui.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAGUI_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
