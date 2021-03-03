cmake_minimum_required(VERSION 3.12)
project(SofaExporter LANGUAGES CXX)

find_package(SofaBaseVisual REQUIRED)
sofa_find_package(ZLIB REQUIRED)
sofa_find_package(SofaPython QUIET)

set(SRC_ROOT src/SofaExporter)

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/initSofaExporter.h
    )

set(SOURCE_FILES
    ${SRC_ROOT}/initSofaExporter.cpp
    )

set(EXTRA_FILES
    README.md
    examples/MeshExporter.scn
    examples/STLExporter.scn
    examples/OBJExporter.scn
    )

list(APPEND HEADER_FILES
    ${SRC_ROOT}/BlenderExporter.h
    ${SRC_ROOT}/BlenderExporter.inl
    ${SRC_ROOT}/MeshExporter.h
    ${SRC_ROOT}/OBJExporter.h
    ${SRC_ROOT}/STLExporter.h
    ${SRC_ROOT}/VTKExporter.h
    ${SRC_ROOT}/WriteState.h
    ${SRC_ROOT}/WriteState.inl
    ${SRC_ROOT}/WriteTopology.h
    ${SRC_ROOT}/WriteTopology.inl
    )

list(APPEND SOURCE_FILES
    ${SRC_ROOT}/BlenderExporter.cpp
    ${SRC_ROOT}/MeshExporter.cpp
    ${SRC_ROOT}/OBJExporter.cpp
    ${SRC_ROOT}/STLExporter.cpp
    ${SRC_ROOT}/VTKExporter.cpp
    ${SRC_ROOT}/WriteState.cpp
    ${SRC_ROOT}/WriteTopology.cpp
    )

if(SofaPython_FOUND)
    list(APPEND HEADER_FILES
        ${SRC_ROOT}/bindings/Binding_OBJExporter.h
        ${SRC_ROOT}/bindings/Binding_STLExporter.h
    )
    list(APPEND SOURCE_FILES
        ${SRC_ROOT}/bindings/Binding_OBJExporter.cpp
        ${SRC_ROOT}/bindings/Binding_STLExporter.cpp
    )
else()
    message(STATUS "SofaPython disabled, will not compile python bindings.")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseVisual)
target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    sofa_install_libraries(TARGETS ZLIB::ZLIB)
endif()
if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaPython)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME SofaExporter
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaExporter"
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAEXPORTER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAEXPORTER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaExporter_test)
endif()
