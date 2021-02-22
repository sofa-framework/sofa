cmake_minimum_required(VERSION 3.12)
project(Sofa.Topology LANGUAGES CXX)

set(SOFATOPOLOGYSRC_ROOT "src/sofa/topology")

set(HEADER_FILES
    ${SOFATOPOLOGYSRC_ROOT}/config.h.in
    ${SOFATOPOLOGYSRC_ROOT}/init.h
)

set(SOURCE_FILES
    ${SOFATOPOLOGYSRC_ROOT}/init.cpp
)

# Compatibility files
set (COMPATSRC_ROOT "compat/")
set(COMPAT_HEADER_FILES
)

find_package(Sofa.Config REQUIRED)
# find_package(Sofa.Type REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${COMPAT_HEADER_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Type)

target_include_directories(${PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${COMPATSRC_ROOT}>
                           $<INSTALL_INTERFACE:include/SofaFramework/compat>)
source_group("compat" FILES ${COMPAT_HEADER_FILES})

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )
