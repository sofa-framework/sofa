cmake_minimum_required(VERSION 3.12)
project(SofaNewmat LANGUAGES CXX)

find_package(SofaFramework REQUIRED)

set(SOFANEWMAT_SRC "src/${PROJECT_NAME}")


set(HEADER_FILES
    ${SOFANEWMAT_SRC}/config.h.in
    ${SOFANEWMAT_SRC}/initSofaNewmat.h
)

set(SOURCE_FILES
    ${SOFANEWMAT_SRC}/initSofaNewmat.cpp
)

add_library(${PROJECT_NAME} SHARED  ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.SimulationCore)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)