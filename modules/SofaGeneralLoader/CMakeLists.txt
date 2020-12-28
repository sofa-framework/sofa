cmake_minimum_required(VERSION 3.12)
project(SofaGeneralLoader LANGUAGES CXX)

set(SOFAGENERALLOADER_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALLOADER_SRC}/config.h.in
    ${SOFAGENERALLOADER_SRC}/initSofaGeneralLoader.h
    )

set(SOURCE_FILES
    ${SOFAGENERALLOADER_SRC}/initSofaGeneralLoader.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALLOADER_SRC}/ReadState.h
    ${SOFAGENERALLOADER_SRC}/ReadState.inl
    ${SOFAGENERALLOADER_SRC}/ReadTopology.h
    ${SOFAGENERALLOADER_SRC}/ReadTopology.inl
    ${SOFAGENERALLOADER_SRC}/GIDMeshLoader.h
    ${SOFAGENERALLOADER_SRC}/GridMeshCreator.h
    ${SOFAGENERALLOADER_SRC}/InputEventReader.h
    ${SOFAGENERALLOADER_SRC}/MeshGmshLoader.h
    ${SOFAGENERALLOADER_SRC}/MeshOffLoader.h
    ${SOFAGENERALLOADER_SRC}/MeshSTLLoader.h
    ${SOFAGENERALLOADER_SRC}/MeshTrianLoader.h
    ${SOFAGENERALLOADER_SRC}/MeshXspLoader.h
    ${SOFAGENERALLOADER_SRC}/OffSequenceLoader.h
    ${SOFAGENERALLOADER_SRC}/SphereLoader.h
    ${SOFAGENERALLOADER_SRC}/StringMeshCreator.h
    ${SOFAGENERALLOADER_SRC}/VoxelGridLoader.h
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALLOADER_SRC}/ReadState.cpp
    ${SOFAGENERALLOADER_SRC}/ReadTopology.cpp
    ${SOFAGENERALLOADER_SRC}/GIDMeshLoader.cpp
    ${SOFAGENERALLOADER_SRC}/GridMeshCreator.cpp
    ${SOFAGENERALLOADER_SRC}/InputEventReader.cpp
    ${SOFAGENERALLOADER_SRC}/MeshGmshLoader.cpp
    ${SOFAGENERALLOADER_SRC}/MeshOffLoader.cpp
    ${SOFAGENERALLOADER_SRC}/MeshSTLLoader.cpp
    ${SOFAGENERALLOADER_SRC}/MeshTrianLoader.cpp
    ${SOFAGENERALLOADER_SRC}/MeshXspLoader.cpp
    ${SOFAGENERALLOADER_SRC}/OffSequenceLoader.cpp
    ${SOFAGENERALLOADER_SRC}/SphereLoader.cpp
    ${SOFAGENERALLOADER_SRC}/StringMeshCreator.cpp
    ${SOFAGENERALLOADER_SRC}/VoxelGridLoader.cpp
    )

sofa_find_package(ZLIB REQUIRED BOTH_SCOPES)

find_package(SofaSimulation REQUIRED)
find_package(SofaFramework REQUIRED) # SofaCore SofaHelper

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaHelper SofaSimulationCore)
target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    sofa_install_libraries(TARGETS ZLIB::ZLIB)
endif()


sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAGENERALLOADER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAGENERALLOADER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
