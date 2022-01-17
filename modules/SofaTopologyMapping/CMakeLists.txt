cmake_minimum_required(VERSION 3.12)
project(SofaTopologyMapping LANGUAGES CXX)

set(SOFATOPOLOGYMAPPING_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFATOPOLOGYMAPPING_SRC}/config.h.in
    ${SOFATOPOLOGYMAPPING_SRC}/initSofaTopologyMapping.h
)

set(SOURCE_FILES
    ${SOFATOPOLOGYMAPPING_SRC}/initSofaTopologyMapping.cpp
)

sofa_find_package(SofaBaseTopology REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
