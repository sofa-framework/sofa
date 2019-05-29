cmake_minimum_required(VERSION 3.1)
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


add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/extlibs/TinyXML>")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${TINYXML_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

include(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework/SofaMacros.cmake)
sofa_generate_package(NAME TinyXML VERSION ${TINYXML_VERSION} TARGETS ${PROJECT_NAME} INCLUDE_INSTALL_DIR "extlibs/TinyXML/tinyxml")
