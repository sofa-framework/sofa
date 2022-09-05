cmake_minimum_required(VERSION 3.12)
project(tinyxml VERSION 2.6.2)

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


add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})

include(SofaMacros)
sofa_create_package_with_targets(
    PACKAGE_NAME TinyXML
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "."
    INCLUDE_INSTALL_DIR "extlibs/TinyXML"
    )
