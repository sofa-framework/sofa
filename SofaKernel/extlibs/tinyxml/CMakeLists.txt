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

include(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework/SofaMacros.cmake)
sofa_generate_package(
    NAME TinyXML
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "."
    INCLUDE_INSTALL_DIR "extlibs/TinyXML"
    )
