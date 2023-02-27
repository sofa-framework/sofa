cmake_minimum_required(VERSION 3.12)
project(Sofa.Geometry LANGUAGES CXX)

set(SOFAGEOMETRYSRC_ROOT "src/sofa/geometry")

set(HEADER_FILES
    ${SOFAGEOMETRYSRC_ROOT}/config.h.in
    ${SOFAGEOMETRYSRC_ROOT}/init.h
    ${SOFAGEOMETRYSRC_ROOT}/Edge.h
    ${SOFAGEOMETRYSRC_ROOT}/ElementInfo.h
    ${SOFAGEOMETRYSRC_ROOT}/ElementType.h
    ${SOFAGEOMETRYSRC_ROOT}/Hexahedron.h
    ${SOFAGEOMETRYSRC_ROOT}/Pentahedron.h
    ${SOFAGEOMETRYSRC_ROOT}/Point.h
    ${SOFAGEOMETRYSRC_ROOT}/Pyramid.h
    ${SOFAGEOMETRYSRC_ROOT}/Quad.h
    ${SOFAGEOMETRYSRC_ROOT}/Tetrahedron.h
    ${SOFAGEOMETRYSRC_ROOT}/Triangle.h

    ${SOFAGEOMETRYSRC_ROOT}/proximity/PointTriangle.h
    ${SOFAGEOMETRYSRC_ROOT}/proximity/SegmentTriangle.h
    ${SOFAGEOMETRYSRC_ROOT}/proximity/TriangleTriangle.h
)

set(SOURCE_FILES
    ${SOFAGEOMETRYSRC_ROOT}/init.cpp
    ${SOFAGEOMETRYSRC_ROOT}/ElementInfo.cpp
)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Type REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config Sofa.Type)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_GEOMETRY_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_GEOMETRY_BUILD_TESTS)
    add_subdirectory(test)
endif()
