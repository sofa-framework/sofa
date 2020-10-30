cmake_minimum_required(VERSION 3.12)
project(SofaMiscExtra LANGUAGES CXX)

find_package(SofaMiscTopology REQUIRED)
find_package(SofaGeneral REQUIRED)

set(SOFAMISCEXTRA_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCEXTRA_SRC}/initSofaMiscExtra.h
    ${SOFAMISCEXTRA_SRC}/config.h.in
    )

set(SOURCE_FILES
    ${SOFAMISCEXTRA_SRC}/initSofaMiscExtra.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAMISCEXTRA_SRC}/MeshTetraStuffing.h
    )
list(APPEND SOURCE_FILES
    ${SOFAMISCEXTRA_SRC}/MeshTetraStuffing.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGeneralMeshCollision SofaMiscTopology SofaTopologyMapping)

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
cmake_dependent_option(SOFAMISCEXTRA_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCEXTRA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscExtra_test)
endif()
