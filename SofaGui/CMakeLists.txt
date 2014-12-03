cmake_minimum_required(VERSION 2.8.11)
set(SOLUTION_NAME "SofaGui")
project(${SOLUTION_NAME})

#temporary until we move the sub projects
set( SofaGui_ACTUAL_SOURCE_DIR ${SofaGui_SOURCE_DIR}/../applications/sofa/gui)
set( SofaProjects_ACTUAL_SOURCE_DIR ${SofaGui_SOURCE_DIR}/../applications/projects)

##
## SofaGui version
##
set( SOFAGUI_MAJOR_VERSION 0 )
set( SOFAGUI_MINOR_VERSION 0 )
set( SOFAGUI_BUILD_VERSION 1 )

set( SOFAGUI_VERSION ${SOFAGUI_MAJOR_VERSION}.${SOFAGUI_MINOR_VERSION}.${SOFAGUI_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(SOFAGUI_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFAGUI_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFAGUI_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAGUI_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaGui)

# Offer the user the choice of overriding the installation directories
set(SOFAGUI_INSTALL_LIB_DIR     lib     					  CACHE PATH "Installation directory for libraries")
set(SOFAGUI_INSTALL_BIN_DIR     bin     					  CACHE PATH "Installation directory for executables")
set(SOFAGUI_INSTALL_INCLUDE_DIR include 					  CACHE PATH "Installation directory for header files")
set(SOFAGUI_INSTALL_CMAKE_DIR   ${SOFAGUI_INSTALL_LIB_DIR}/cmake/SofaGui)

## SofaGui dependencies
##
find_package(SofaGeneral REQUIRED)
set(SOFAGUI_DEPENDENCY_DIR ${SofaGeneral_INCLUDE_DIRS})

find_package(Qt4 COMPONENTS qtcore qtgui qtopengl qtxml qt3support)

set(SOFA_HAVE_QT ${Qt4_FOUND} )
set(SOFA_QT4 ${Qt4_FOUND} )

#set(SOFAGUI_TARGETS SofaGuiCommon SofaGuiQt SofaGuiQTViewer SofaGuiGlut SofaGuiMain)

set(SOFAGUI_TARGETS SofaGuiCommon)

add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/SofaGuiCommon ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiCommon)

if(SOFA_HAVE_GLUT)
    set(SOFA_GUI_GLUT 1)
    list(APPEND SOFAGUI_TARGETS SofaGuiGlut)
    add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/glut          ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiGlut)
endif()

if(Qt4_FOUND)
    set(SOFA_GUI_QT 1)

    list(APPEND SOFAGUI_TARGETS SofaGuiQt)
    add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/qt            ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQt)
    set(SOFA_GUI_QTVIEWER 1)
    
    #add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/qt/viewer/qt  ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQTViewer)
    #add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/qt/viewer/qgl ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiQTQGLViewer)
endif(Qt4_FOUND)


list(APPEND SOFAGUI_TARGETS SofaGuiMain)
add_subdirectory(${SofaGui_ACTUAL_SOURCE_DIR}/SofaGuiMain ${CMAKE_CURRENT_BINARY_DIR}/SofaGuiMain)

# applications
if(Qt4_FOUND)
    add_subdirectory(${SofaProjects_ACTUAL_SOURCE_DIR}/Modeler            ${CMAKE_CURRENT_BINARY_DIR}/Modeler)
endif(Qt4_FOUND)
add_subdirectory(${SofaProjects_ACTUAL_SOURCE_DIR}/runSofa            ${CMAKE_CURRENT_BINARY_DIR}/runSofa)

foreach(TARGET ${SOFAGUI_TARGETS})
  # set IDE project filter
  if(MSVC)
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaGui")
  endif()
endforeach()

configure_file( ${SofaGui_SOURCE_DIR}/SofaGui.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaGui.h )

install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaGui.h
         DESTINATION "${SOFAGUI_INSTALL_INCLUDE_DIR}/sofa")

include(CMakePackageConfigHelpers)

# SofaGuiConfigVersion.cmake file (same for build tree and install)
write_basic_package_version_file(${SOFAGUI_BUILD_CMAKE_DIR}/SofaGuiConfigVersion.cmake
                                 VERSION ${SOFAGUI_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${SOFAGUI_BUILD_CMAKE_DIR}/SofaGuiConfigVersion.cmake
        DESTINATION ${SOFAGUI_INSTALL_CMAKE_DIR})

# SofaGuiConfig.cmake (build tree)
set(SOFAGUI_INCLUDE_DIR "${SOFAGUI_BUILD_INCLUDE_DIR}")
configure_package_config_file(SofaGuiConfig.cmake.in
                              ${SOFAGUI_BUILD_CMAKE_DIR}/SofaGuiConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFAGUI_INCLUDE_DIR)

# SofaGuiConfig.cmake (installed)
set(SOFAGUI_INCLUDE_DIR "${SOFAGUI_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaGuiConfig.cmake.in
                              ${SOFAGUI_BUILD_CMAKE_DIR}/SofaGuiConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAGUI_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAGUI_INCLUDE_DIR )

install(FILES ${SOFAGUI_BUILD_CMAKE_DIR}/SofaGuiConfigInstall.cmake
        DESTINATION ${SOFAGUI_INSTALL_CMAKE_DIR}
        RENAME SofaGuiConfig.cmake)

# SofaGuiTargets.cmake (installed)
install(EXPORT SofaGuiTargets
        DESTINATION ${SOFAGUI_INSTALL_CMAKE_DIR} )
