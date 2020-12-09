cmake_minimum_required(VERSION 3.12)
project(SofaGeneralTopology LANGUAGES CXX)

set(SOFAGENERALTOPOLOGY_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALTOPOLOGY_SRC}/config.h.in
    ${SOFAGENERALTOPOLOGY_SRC}/initSofaGeneralTopology.h
    )

set(SOURCE_FILES
    ${SOFAGENERALTOPOLOGY_SRC}/initSofaGeneralTopology.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALTOPOLOGY_SRC}/CubeTopology.h
    ${SOFAGENERALTOPOLOGY_SRC}/CylinderGridTopology.h
    ${SOFAGENERALTOPOLOGY_SRC}/SphereGridTopology.h
    ${SOFAGENERALTOPOLOGY_SRC}/SphereQuadTopology.h
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALTOPOLOGY_SRC}/CubeTopology.cpp
    ${SOFAGENERALTOPOLOGY_SRC}/CylinderGridTopology.cpp
    ${SOFAGENERALTOPOLOGY_SRC}/SphereGridTopology.cpp
    ${SOFAGENERALTOPOLOGY_SRC}/SphereQuadTopology.cpp
    )

find_package(SofaBase REQUIRED) # SofaBaseTopology

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

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAGENERALTOPOLOGY_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAGENERALTOPOLOGY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
