cmake_minimum_required(VERSION 3.12)
project(Sofa.GUI LANGUAGES CXX)

set(SOFAGUI_SOURCE_DIR "src/sofa/gui")

set(SOFAGUI_TARGETS)
set(SOFAGUI_MISSINGTARGETS)

sofa_add_subdirectory(module Component ${PROJECT_NAME}.Component ON)
if(TARGET ${PROJECT_NAME}.Component)
    list(APPEND SOFAGUI_TARGETS ${PROJECT_NAME}.Component)
else()
    list(APPEND SOFAGUI_MISSINGTARGETS ${PROJECT_NAME}.Component)
endif()

sofa_add_subdirectory(library Common ${PROJECT_NAME}.Common ON)
if(TARGET ${PROJECT_NAME}.Common)
    list(APPEND SOFAGUI_TARGETS ${PROJECT_NAME}.Common)
else()
    list(APPEND SOFAGUI_MISSINGTARGETS ${PROJECT_NAME}.Common)
endif()

set(SOFAGUI_DIRS Batch Qt)
foreach(dir ${SOFAGUI_DIRS})
    sofa_add_subdirectory(plugin ${dir} ${PROJECT_NAME}.${dir} ON)
    if(TARGET ${PROJECT_NAME}.${dir})
        list(APPEND SOFAGUI_TARGETS ${PROJECT_NAME}.${dir})
    else()
        list(APPEND SOFAGUI_MISSINGTARGETS ${PROJECT_NAME}.${dir})
    endif()
endforeach()

# HeadlessRecorder is still an "optional" dependency of Sofa.GUI
sofa_add_subdirectory(plugin HeadlessRecorder ${PROJECT_NAME}.HeadlessRecorder OFF)
if(TARGET ${PROJECT_NAME}.HeadlessRecorder)
    list(APPEND SOFAGUI_TARGETS ${PROJECT_NAME}.HeadlessRecorder)
    set(SOFA_GUI_HAVE_SOFA_GUI_HEADLESSRECORDER 1)
endif()

option(LIBRARY_SOFA_GUI "Build the Sofa.Gui library." ON)

if(LIBRARY_SOFA_GUI)
    if(SOFAGUI_MISSINGTARGETS)
        message("${PROJECT_NAME}: package and library will not be created because some dependencies are missing or disabled: ${SOFAGUI_MISSINGTARGETS}")
        return()
    endif()

    set(HEADER_FILES
        ${SOFAGUI_SOURCE_DIR}/config.h.in
        ${SOFAGUI_SOURCE_DIR}/init.h
    )
    set(SOURCE_FILES
        ${SOFAGUI_SOURCE_DIR}/init.cpp
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

endif()
