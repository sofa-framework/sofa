cmake_minimum_required(VERSION 2.8.11)
set(SOLUTION_NAME "SofaGui")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set(SofaGui_ACTUAL_SOURCE_DIR ${SofaGui_SOURCE_DIR}/../applications/sofa/gui)
set(SofaProjects_ACTUAL_SOURCE_DIR ${SofaGui_SOURCE_DIR}/../applications/projects)

##
## SofaGui version
##
set(SOFAGUI_MAJOR_VERSION 0)
set(SOFAGUI_MINOR_VERSION 0)
set(SOFAGUI_BUILD_VERSION 1)

set(SOFAGUI_VERSION ${SOFAGUI_MAJOR_VERSION}.${SOFAGUI_MINOR_VERSION}.${SOFAGUI_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

# Offer the user the choice of overriding the installation directories
set(SOFAGUI_INSTALL_LIB_DIR lib 					 CACHE PATH "Installation directory for libraries")
set(SOFAGUI_INSTALL_BIN_DIR bin 					 CACHE PATH "Installation directory for executables")
set(SOFAGUI_INSTALL_INCLUDE_DIR include 					 CACHE PATH "Installation directory for header files")
set(SOFAGUI_INSTALL_CMAKE_DIR ${SOFAGUI_INSTALL_LIB_DIR}/cmake/SofaGui)

## SofaGui dependencies
##
find_package(SofaGeneral REQUIRED)
find_package(SofaMisc REQUIRED)
find_package(SofaAdvanced REQUIRED)
find_package(QGLViewer REQUIRED)
set(SOFAGUI_DEPENDENCY_DIR ${SofaGeneral_INCLUDE_DIRS} ${SofaMisc_INCLUDE_DIRS} ${SofaAdvanced_INCLUDE_DIRS})

set(QT_USE_IMPORTED_TARGETS 1)
set(QT4_NO_LINK_QTMAIN 1)
find_package(Qt4 COMPONENTS qtcore qtgui qtopengl qtxml qt3support)
set(SOFAGUI_DEPENDENCY_DIR ${SOFAGUI_DEPENDENCY_DIR} ${QT_INCLUDES})

if(WIN32)
    file(GLOB QT4_DLLS "${QT_BINARY_DIR}/*.dll")
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        file(COPY ${QT4_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
    endforeach()
endif()
install(FILES ${QT4_DLLS} DESTINATION ${SOFAGUI_INSTALL_BIN_DIR})

set(SOFA_HAVE_QT ${Qt4_FOUND})
set(SOFA_QT4 ${Qt4_FOUND})

set(SOFAGUI_TARGETS SofaGuiCommon)

add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/SofaGuiCommon ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiCommon)

# Glut GUI
option(SOFAGUI_GLUT "Compile the Glut GUI" ON)
if(SOFAGUI_GLUT)
    set(SOFA_GUI_GLUT 1)        # Macro in SofaGuiConfig.h
    list(APPEND SOFAGUI_TARGETS SofaGuiGlut)
    add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/glut ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiGlut)
endif()

# Qt GUI
option(SOFAGUI_QT "Compile the Qt GUI" ON)
if(SOFAGUI_QT)
    set(SOFA_GUI_QT 1)          # Macro in SofaGuiConfig.h
    list(APPEND SOFAGUI_TARGETS SofaGuiQt)
    add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/qt ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQt)

    # SOFA_GUI_INTERACTION
    option(SOFAGUI_QT_INTERACTION "Enable interaction mode for the Qt GUI, whatever that means" ON)
    if(SOFAGUI_QT_INTERACTION)
        set(SOFA_GUI_INTERACTION 1) # Macro in SofaGuiConfig.h
    endif()

    # QtViewer
    option(SOFAGUI_QTVIEWER "Compile the QtViewer for the Qt GUI" ON)
    if(SOFA_GUI_QTVIEWER)
        set(SOFA_GUI_QTVIEWER 1) # Macro in SofaGuiConfig.h
    endif()

    # QglViewer
    option(SOFAGUI_QGLVIEWER "Compile the QGLViewer for the Qt GUI" ON)
    if(SOFA_GUI_QGLVIEWER)
        set(SOFA_GUI_QGLVIEWER 1) # Macro in SofaGuiConfig.h
    endif()
endif()


list(APPEND SOFAGUI_TARGETS SofaGuiMain)
add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/SofaGuiMain ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiMain)

# applications
if(Qt4_FOUND)
    add_subdirectory(${SofaProjects_ACTUAL_SOURCE_DIR}/Modeler ${CMAKE_CURRENT_BINARY_DIR}/Modeler)
    add_subdirectory(${SofaProjects_ACTUAL_SOURCE_DIR}/runSofa ${CMAKE_CURRENT_BINARY_DIR}/runSofa)
    list(APPEND SOFAGUI_TARGETS Modeler runSofa)
endif(Qt4_FOUND)

foreach(TARGET ${SOFAGUI_TARGETS})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaGui") # IDE folder
endforeach()

configure_file(SofaGuiConfig.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/SofaGui/SofaGuiConfig.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaGui/SofaGuiConfig.h
        DESTINATION "${SOFAGUI_INSTALL_INCLUDE_DIR}/SofaGui")

include(CMakePackageConfigHelpers)

# SofaGuiConfigVersion.cmake file (same for build tree and install)
write_basic_package_version_file(SofaGuiConfigVersion.cmake
                                 VERSION ${SOFAGUI_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiConfigVersion.cmake
        DESTINATION ${SOFAGUI_INSTALL_CMAKE_DIR})

# SofaGuiConfig.cmake (build tree)
set(SOFAGUI_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(SofaGuiConfig.cmake.in
                              SofaGuiConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFAGUI_INCLUDE_DIR)

# SofaGuiConfig.cmake (installed)
set(SOFAGUI_INCLUDE_DIR "${SOFAGUI_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaGuiConfig.cmake.in
                              SofaGuiConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAGUI_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAGUI_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiConfigInstall.cmake
        DESTINATION ${SOFAGUI_INSTALL_CMAKE_DIR}
        RENAME SofaGuiConfig.cmake)

# SofaGuiTargets.cmake (installed)
install(EXPORT SofaGuiTargets
        DESTINATION ${SOFAGUI_INSTALL_CMAKE_DIR})
