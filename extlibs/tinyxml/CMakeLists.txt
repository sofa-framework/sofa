# Copyright (C) 2012, 2013 iCub Facility, Istituto Italiano di Tecnologia
# Author: Daniele E. Domenichelli <daniele.domenichelli@iit.it>
# CopyPolicy: Released under the terms of the LGPLv2.1 or later, see LGPL.TXT

cmake_minimum_required(VERSION 2.8.12)
project(tinyxml)


set(TINYXML_MAJOR_VERSION 2)
set(TINYXML_MINOR_VERSION 6)
set(TINYXML_PATCH_VERSION 2)
set(TINYXML_VERSION ${TINYXML_MAJOR_VERSION}.${TINYXML_MINOR_VERSION}.${TINYXML_PATCH_VERSION})

# Offer the user the choice of overriding the installation directories
set(TINYXML_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(TINYXML_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(TINYXML_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
set(TINYXML_INSTALL_CMAKE_DIR lib/cmake/TinyXML CACHE PATH "Installation directory for CMake files")

set(tinyxml_SRCS
    tinystr.cpp
    tinyxml.cpp
    tinyxmlerror.cpp
    tinyxmlparser.cpp)

set(tinyxml_HDRS
    tinystr.h
    tinyxml.h)


add_library(tinyxml SHARED ${tinyxml_SRCS} ${tinyxml_HDRS})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_include_directories(tinyxml PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_compile_options(tinyxml PUBLIC "-DTIXML_USE_STL" "-DTI")

set_target_properties(tinyxml PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(tinyxml PROPERTIES VERSION ${TINYXML_VERSION})
set_target_properties(tinyxml PROPERTIES PUBLIC_HEADER "${tinyxml_HDRS}")

install(TARGETS tinyxml
        EXPORT TinyXMLTargets
        RUNTIME DESTINATION "${TINYXML_INSTALL_BIN_DIR}" COMPONENT TinyXML_libraries
        LIBRARY DESTINATION "${TINYXML_INSTALL_LIB_DIR}" COMPONENT TinyXML_libraries
        ARCHIVE DESTINATION "${TINYXML_INSTALL_LIB_DIR}" COMPONENT TinyXML_libraries
        PUBLIC_HEADER DESTINATION "${TINYXML_INSTALL_INCLUDE_DIR}" COMPONENT TinyXML_headers)

include(CMakePackageConfigHelpers)

# TinyXMLConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(TinyXMLConfigVersion.cmake
                                 VERSION ${TINYXML_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TinyXMLConfigVersion.cmake
        DESTINATION ${TINYXML_INSTALL_CMAKE_DIR})


# TinyXMLConfig.cmake (build tree)
set(TINYXML_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(TinyXMLConfig.cmake.in
                              TinyXMLConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS TINYXML_INCLUDE_DIR)

# TinyXMLConfig.cmake (installed)
set(TINYXML_INCLUDE_DIR "${TINYXML_INSTALL_INCLUDE_DIR}")
configure_package_config_file(TinyXMLConfig.cmake.in
                              TinyXMLConfigInstall.cmake
                              INSTALL_DESTINATION ${TINYXML_INSTALL_CMAKE_DIR}
                              PATH_VARS TINYXML_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TinyXMLConfigInstall.cmake
        DESTINATION ${TINYXML_INSTALL_CMAKE_DIR}
        RENAME TinyXMLConfig.cmake)

# TinyXMLTargets.cmake (installed)
install(EXPORT TinyXMLTargets
        DESTINATION ${TINYXML_INSTALL_CMAKE_DIR})
