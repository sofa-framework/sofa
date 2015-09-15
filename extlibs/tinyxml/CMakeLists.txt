cmake_minimum_required(VERSION 2.8.12)
project(tinyxml)


set(TINYXML_VERSION 2.6.2)

set(SOURCE_FILES
    tinystr.cpp
    tinyxml.cpp
    tinyxmlerror.cpp
    tinyxmlparser.cpp
)

set(HEADER_FILES
    tinystr.h
    tinyxml.h
)


add_library(tinyxml SHARED ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_include_directories(tinyxml PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
set_target_properties(tinyxml PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(tinyxml PROPERTIES VERSION ${TINYXML_VERSION})
set_target_properties(tinyxml PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(TARGETS tinyxml
        EXPORT TinyXMLTargets
        RUNTIME DESTINATION bin COMPONENT TinyXML_libraries
        LIBRARY DESTINATION lib COMPONENT TinyXML_libraries
        ARCHIVE DESTINATION lib COMPONENT TinyXML_libraries
        PUBLIC_HEADER DESTINATION include COMPONENT TinyXML_headers)


include(CMakePackageConfigHelpers)

## TinyXMLConfigVersion.cmake
write_basic_package_version_file(TinyXMLConfigVersion.cmake
                                 VERSION ${TINYXML_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TinyXMLConfigVersion.cmake
        DESTINATION lib/cmake/TinyXML)

### TinyXMLConfig.cmake
## Build tree
set(TINYXML_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(TinyXMLConfig.cmake.in
                              TinyXMLConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS TINYXML_INCLUDE_DIR)
## Install tree
set(TINYXML_INCLUDE_DIR include)
configure_package_config_file(TinyXMLConfig.cmake.in
                              InstalledTinyXMLConfig.cmake
                              INSTALL_DESTINATION lib/cmake/TinyXML
                              PATH_VARS TINYXML_INCLUDE_DIR)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/InstalledTinyXMLConfig.cmake
        DESTINATION lib/cmake/TinyXML
        RENAME TinyXMLConfig.cmake)

## TinyXMLTargets.cmake
install(EXPORT TinyXMLTargets DESTINATION lib/cmake/TinyXML)
