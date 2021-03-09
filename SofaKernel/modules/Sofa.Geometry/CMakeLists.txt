cmake_minimum_required(VERSION 3.12)
project(Sofa.Geometry LANGUAGES CXX)

set(SOFAGEOMETRYSRC_ROOT "src/sofa/geometry")

set(HEADER_FILES
    ${SOFAGEOMETRYSRC_ROOT}/config.h.in
    ${SOFAGEOMETRYSRC_ROOT}/init.h
    ${SOFAGEOMETRYSRC_ROOT}/ElementType.h
    ${SOFAGEOMETRYSRC_ROOT}/ElementInfo.h
    ${SOFAGEOMETRYSRC_ROOT}/Point.h
    ${SOFAGEOMETRYSRC_ROOT}/Edge.h
    ${SOFAGEOMETRYSRC_ROOT}/Triangle.h
    ${SOFAGEOMETRYSRC_ROOT}/Quad.h
    ${SOFAGEOMETRYSRC_ROOT}/Tetrahedron.h
    ${SOFAGEOMETRYSRC_ROOT}/Pentahedron.h
    ${SOFAGEOMETRYSRC_ROOT}/Pyramid.h
    ${SOFAGEOMETRYSRC_ROOT}/Hexahedron.h
)

set(SOURCE_FILES
    ${SOFAGEOMETRYSRC_ROOT}/init.cpp
    ${SOFAGEOMETRYSRC_ROOT}/ElementInfo.cpp
)

find_package(Sofa.Config REQUIRED)
# find_package(Sofa.Type REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Type)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )
