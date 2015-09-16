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

include(../../SofaFramework/SofaPackageConfigHelpers.cmake)
sofa_create_package(TinyXML 2.6.2 tinyxml "")
