cmake_minimum_required(VERSION 3.1)
project(SofaExporter VERSION 1.0 LANGUAGES CXX)

find_package(SofaSimulation REQUIRED)
find_package(SofaBase REQUIRED) # SofaBaseVisual
sofa_find_package(ZLIB REQUIRED)
sofa_find_package(SofaPython QUIET)

set(SRC_ROOT src/SofaExporter)

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/initExporter.h
    )

set(SOURCE_FILES
    ${SRC_ROOT}/initExporter.cpp
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
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree SofaBaseVisual)
target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    sofa_install_libraries(TARGETS ZLIB::ZLIB)
endif()
if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaPython)
endif()
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_EXPORTER")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

sofa_generate_package(
    NAME SofaExporter
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "${SRC_ROOT}"
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
