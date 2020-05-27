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
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../applications>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
endforeach()


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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUI")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "${SOFAGUI_SRC}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
