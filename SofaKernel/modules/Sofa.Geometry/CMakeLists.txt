cmake_minimum_required(VERSION 3.12)
project(Sofa.Geometry LANGUAGES CXX)

set(SOFAGEOMETRYSRC_ROOT "src/sofa/geometry")

set(HEADER_FILES
    ${SOFAGEOMETRYSRC_ROOT}/config.h.in
    ${SOFAGEOMETRYSRC_ROOT}/init.h
)

set(SOURCE_FILES
    ${SOFAGEOMETRYSRC_ROOT}/init.cpp
)

find_package(Sofa.Config REQUIRED)
# find_package(Sofa.Type REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Type Sofa.Topology SofaHelper)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )
