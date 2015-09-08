cmake_minimum_required(VERSION 2.8.11)

project(SofaGui)
set(SOFAGUI_VERSION 0.0.1)

## Temporary
set(SofaGui_ACTUAL_SOURCE_DIR ${SofaGui_SOURCE_DIR}/../applications/sofa/gui)
set(SofaProjects_ACTUAL_SOURCE_DIR ${SofaGui_SOURCE_DIR}/../applications/projects)


### SofaGui dependencies

# This variable is used in other CMakeLists.txt files (SofaGuiCommon, SofaGuiMain...)
set(SOFAGUI_DEPENDENCY_DIR "")

## Sofa
find_package(SofaGeneral REQUIRED)
list(APPEND SOFAGUI_DEPENDENCY_DIR ${SofaGeneral_INCLUDE_DIRS})
find_package(SofaMisc REQUIRED)
list(APPEND SOFAGUI_DEPENDENCY_DIR ${SofaMisc_INCLUDE_DIRS})
find_package(SofaAdvanced REQUIRED)
list(APPEND SOFAGUI_DEPENDENCY_DIR ${SofaAdvanced_INCLUDE_DIRS})

## Qt4
set(QT_USE_IMPORTED_TARGETS 1)
set(QT4_NO_LINK_QTMAIN 1)
find_package(Qt4 COMPONENTS qtcore qtgui qtopengl qtxml qt3support)
list(APPEND SOFAGUI_DEPENDENCY_DIR ${QT_INCLUDES})
if(WIN32)
    file(GLOB QT4_DLLS "${QT_BINARY_DIR}/*.dll")
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        file(COPY ${QT4_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
    endforeach()
endif()
install(FILES ${QT4_DLLS} DESTINATION bin)

## Thoses variables are used in SofaGuiConfig.cmake.in
set(SOFA_HAVE_QT ${Qt4_FOUND})
set(SOFA_QT4 ${Qt4_FOUND})
set(SOFAGUI_TARGETS "")


### Sub-projects

## SofaGuiCommon
add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/SofaGuiCommon ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiCommon)
list(APPEND SOFAGUI_TARGETS SofaGuiCommon)

## Glut GUI
option(SOFAGUI_GLUT "Compile the Glut GUI" ON)
if(SOFAGUI_GLUT)
    set(SOFA_GUI_GLUT 1) # Used in SofaGuiConfig.*
    list(APPEND SOFAGUI_TARGETS SofaGuiGlut)
    add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/glut ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiGlut)
endif()

## Qt GUI
option(SOFAGUI_QT "Compile the Qt GUI" ON)
if(SOFAGUI_QT)
    set(SOFA_GUI_QT 1) # Used in SofaGuiConfig.*
    list(APPEND SOFAGUI_TARGETS SofaGuiQt)
    add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/qt ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQt)

    ## Qt GUI options, used in the relevant CMakeLists.txt

    # SOFA_GUI_INTERACTION
    option(SOFAGUI_QT_INTERACTION "Enable interaction mode for the Qt GUI, whatever that means
(this will define the SOFA_GUI_INTERACTION macro)." ON)
    if(SOFAGUI_QT_INTERACTION)
        set(SOFA_GUI_INTERACTION 1) # Used in SofaGuiConfig.*
    endif()

    # QtViewer
    option(SOFAGUI_QTVIEWER "Compile the QtViewer for the Qt GUI" ON)
    if(SOFAGUI_QTVIEWER)
        set(SOFA_GUI_QTVIEWER 1) # Used in SofaGuiConfig.*
    endif()

    # QglViewer
    option(SOFAGUI_QGLVIEWER "Compile the QGLViewer for the Qt GUI" ON)
    if(SOFAGUI_QGLVIEWER)
        set(SOFA_GUI_QGLVIEWER 1) # Used in SofaGuiConfig.*
    endif()
endif()

## SofaGuiMain
list(APPEND SOFAGUI_TARGETS SofaGuiMain)
add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/SofaGuiMain ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiMain)

## GUI Applications
option(SOFAGUI_RUNSOFA "Build the runSofa application" ON)
if(SOFAGUI_RUNSOFA)
    add_subdirectory(${SofaProjects_ACTUAL_SOURCE_DIR}/runSofa ${CMAKE_CURRENT_BINARY_DIR}/runSofa)
    list(APPEND SOFAGUI_TARGETS runSofa)
endif()
option(SOFAGUI_MODELER "Build the Modeler application" ON)
if(SOFAGUI_MODELER)
    add_subdirectory(${SofaProjects_ACTUAL_SOURCE_DIR}/Modeler ${CMAKE_CURRENT_BINARY_DIR}/Modeler)
    list(APPEND SOFAGUI_TARGETS Modeler)
endif()


## IDE folder
foreach(TARGET ${SOFAGUI_TARGETS})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaGui")
endforeach()


## SofaGuiConfig.h
configure_file(SofaGuiConfig.h.in SofaGui/SofaGuiConfig.h)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SofaGui/SofaGuiConfig.h"
        DESTINATION "include/SofaGui")


### SofaGuiConfig.cmake

include(CMakePackageConfigHelpers)

## SofaGuiConfigVersion.cmake
write_basic_package_version_file(SofaGuiConfigVersion.cmake VERSION ${SOFAGUI_VERSION} COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SofaGuiConfigVersion.cmake" DESTINATION lib/cmake/SofaGui)

## SofaGuiConfig.cmake (build tree)
set(SOFAGUI_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(SofaGuiConfig.cmake.in SofaGuiConfig.cmake
                              INSTALL_DESTINATION none
                              PATH_VARS SOFAGUI_INCLUDE_DIR)

## SofaGuiConfig.cmake (installed)
set(SOFAGUI_INCLUDE_DIR include)
configure_package_config_file(SofaGuiConfig.cmake.in SofaGuiConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/SofaGui
                              PATH_VARS SOFAGUI_INCLUDE_DIR)
install(EXPORT SofaGuiTargets DESTINATION lib/cmake/SofaGui) # SofaGuiTargets.cmake
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SofaGuiConfigInstall.cmake"
        DESTINATION lib/cmake/SofaGui
        RENAME SofaGuiConfig.cmake)

