cmake_minimum_required(VERSION 3.22)
project(SofaGui)

set(SOFAGUI_TARGETS "")

sofa_find_package(Sofa.GUI.Component REQUIRED)
sofa_find_package(Sofa.GUI.Common REQUIRED)
sofa_find_package(Sofa.GUI.Batch REQUIRED)
sofa_find_package(Sofa.GUI.Qt QUIET)
sofa_find_package(Sofa.GUI.HeadlessRecorder QUIET)

set(SOFAGUI_SRC src/${PROJECT_NAME})
set(HEADER_FILES
    ${SOFAGUI_SRC}/config.h.in
    ${SOFAGUI_SRC}/initSofaGui.h
    )
set(SOURCE_FILES
    ${SOFAGUI_SRC}/initSofaGui.cpp
    )

set(SOFAGUI_TARGETS Sofa.GUI.Component Sofa.GUI.Common Sofa.GUI.Batch)
if(Sofa.GUI.Qt_FOUND)
    list(APPEND SOFAGUI_TARGETS Sofa.GUI.Qt)
endif()
sofa_set_01(SOFAGUI_HAVE_SOFAGUIQT VALUE ${SOFAGUI_HAVE_SOFA_GUI_QT})

if(Sofa.GUI.HeadlessRecorder_FOUND)
    list(APPEND SOFAGUI_TARGETS Sofa.GUI.HeadlessRecorder)
endif()
sofa_set_01(SOFAGUI_HAVE_SOFAHEADLESSRECORDER VALUE ${SOFAGUI_HAVE_SOFA_GUI_HEADLESSRECORDER})

# Keep legacy modules for compatibility
# they merely redirect to the new modules
# legacy (deprecated) module
sofa_add_subdirectory(collection ../modules/SofaGuiCommon SofaGuiCommon ON
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/SofaGuiCommon")
if(TARGET SofaGuiCommon)
    list(APPEND SOFAGUI_TARGETS SofaGuiCommon)
endif()
if(Sofa.GUI.Qt_FOUND)
    sofa_add_subdirectory(collection ../modules/SofaGuiQt SofaGuiQt ON
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQt")
    if(TARGET SofaGuiQt)
        list(APPEND SOFAGUI_TARGETS SofaGuiQt)
    endif()
endif()
if(Sofa.GUI.HeadlessRecorder_FOUND)
    sofa_add_subdirectory(collection ../modules/SofaHeadlessRecorder SofaHeadlessRecorder ON
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/SofaHeadlessRecorder")
    if(TARGET SofaHeadlessRecorder)
        list(APPEND SOFAGUI_TARGETS SofaHeadlessRecorder)
    endif()
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAGUI_TARGETS})

foreach(ui_target ${SOFAGUI_TARGETS})
    set_target_properties(${ui_target} PROPERTIES FOLDER GUI)
endforeach()
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER GUI)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
#    RELOCATABLE "collections" # disabled because we need SofaGui installed at root for backward compat
    )
