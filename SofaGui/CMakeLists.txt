cmake_minimum_required(VERSION 2.8.12)
project(SofaGui)


### Dependencies

## Sofa
find_package(SofaGeneral REQUIRED)
find_package(SofaMisc REQUIRED)
find_package(SofaAdvanced REQUIRED)

## Qt4
set(QT_USE_IMPORTED_TARGETS 1)
set(QT4_NO_LINK_QTMAIN 1)
find_package(Qt4 COMPONENTS qtcore qtgui qtopengl qtxml qt3support)
if(WIN32)
    file(GLOB QT4_DLLS "${QT_BINARY_DIR}/*.dll")
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        file(COPY ${QT4_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
    endforeach()
endif()
install(FILES ${QT4_DLLS} DESTINATION bin)

## Version
set(SOFAGUI_VERSION ${SofaFramework_VERSION})

## Thoses variables are used in SofaGuiConfig.cmake.in
set(SOFA_HAVE_QT ${Qt4_FOUND})
set(SOFA_QT4 ${Qt4_FOUND})
set(SOFAGUI_TARGETS "")


### Sub-projects

## SofaGuiCommon
add_subdirectory(../applications/sofa/gui/SofaGuiCommon ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiCommon)
list(APPEND SOFAGUI_TARGETS SofaGuiCommon)

## Glut GUI
option(SOFAGUI_GLUT "Compile the Glut GUI" ON)
if(SOFAGUI_GLUT)
    set(SOFA_GUI_GLUT 1) # cmakedefine
    list(APPEND SOFAGUI_TARGETS SofaGuiGlut)
    add_subdirectory(../applications/sofa/gui/glut ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiGlut)
endif()

## Qt GUI
option(SOFAGUI_QT "Compile the Qt GUI" ON)
if(SOFAGUI_QT)
    set(SOFA_GUI_QT 1) # cmakedefine
    list(APPEND SOFAGUI_TARGETS SofaGuiQt)
    add_subdirectory(../applications/sofa/gui/qt ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQt)

    ## Qt GUI options, used in the relevant CMakeLists.txt

    ## SOFA_GUI_INTERACTION
    option(SOFAGUI_QT_INTERACTION
"Enable interaction mode for the Qt GUI, whatever that means (this
will define the SOFA_GUI_INTERACTION macro)." OFF)
    if(SOFAGUI_QT_INTERACTION)
        set(SOFA_GUI_INTERACTION 1) # cmakedefine
    endif()

    ## QtViewer
    option(SOFAGUI_QTVIEWER "Compile the QtViewer for the Qt GUI" ON)
    if(SOFAGUI_QTVIEWER)
        set(SOFA_GUI_QTVIEWER 1) # cmakedefine
    endif()

    ## QglViewer
    option(SOFAGUI_QGLVIEWER "Compile the QGLViewer for the Qt GUI" ON)
    if(SOFAGUI_QGLVIEWER)
        set(SOFA_GUI_QGLVIEWER 1) # cmakedefine
    endif()
endif()

## SofaGuiMain
list(APPEND SOFAGUI_TARGETS SofaGuiMain)
add_subdirectory(../applications/sofa/gui/SofaGuiMain ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiMain)

## GUI Applications
option(SOFAGUI_RUNSOFA "Build the runSofa application" ON)
if(SOFAGUI_RUNSOFA)
    add_subdirectory(../applications/projects/runSofa ${CMAKE_CURRENT_BINARY_DIR}/runSofa)
    list(APPEND SOFAGUI_TARGETS runSofa)
endif()
option(SOFAGUI_MODELER "Build the Modeler application" ON)
if(SOFAGUI_MODELER)
    add_subdirectory(../applications/projects/Modeler ${CMAKE_CURRENT_BINARY_DIR}/Modeler)
    list(APPEND SOFAGUI_TARGETS Modeler)
endif()


## IDE folder
foreach(TARGET ${SOFAGUI_TARGETS})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaGui")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../applications>")
    # target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${QT_INCLUDES}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${TARGET} PROPERTIES VERSION ${SOFAGUI_VERSION})
endforeach()


## SofaGuiConfig/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/SofaGui/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/SofaGui/config.h" DESTINATION "include/SofaGui")

## CMake package configurations files
sofa_write_package_config_files(SofaGui ${SOFAGUI_VERSION})
