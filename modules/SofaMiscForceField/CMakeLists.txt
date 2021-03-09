cmake_minimum_required(VERSION 3.12)
project(SofaMiscForceField LANGUAGES CXX)

set(SOFAMISCFORCEFIELD_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCFORCEFIELD_SRC}/config.h.in
    ${SOFAMISCFORCEFIELD_SRC}/initSofaMiscForceField.h
    )

set(SOURCE_FILES
    ${SOFAMISCFORCEFIELD_SRC}/initSofaMiscForceField.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAMISCFORCEFIELD_SRC}/GearSpringForceField.h
    ${SOFAMISCFORCEFIELD_SRC}/GearSpringForceField.inl
    ${SOFAMISCFORCEFIELD_SRC}/MeshMatrixMass.h
    ${SOFAMISCFORCEFIELD_SRC}/MeshMatrixMass.inl
    )
list(APPEND SOURCE_FILES
    ${SOFAMISCFORCEFIELD_SRC}/GearSpringForceField.cpp
    ${SOFAMISCFORCEFIELD_SRC}/MeshMatrixMass.cpp
    )

if(SOFA_WITH_DEPRECATED_COMPONENTS)
    list(APPEND HEADER_FILES
        ${SOFAMISCFORCEFIELD_SRC}/LennardJonesForceField.h
        ${SOFAMISCFORCEFIELD_SRC}/LennardJonesForceField.inl
        )
    list(APPEND SOURCE_FILES
        ${SOFAMISCFORCEFIELD_SRC}/LennardJonesForceField.cpp
        )
endif()

sofa_find_package(SofaFramework REQUIRED) # SofaHelper
sofa_find_package(SofaDeformable REQUIRED)
sofa_find_package(SofaBoundaryCondition REQUIRED)
sofa_find_package(SofaGeneralTopology REQUIRED) 
sofa_find_package(SofaMiscTopology REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDeformable SofaBoundaryCondition SofaMiscTopology SofaGeneralTopology)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
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
cmake_dependent_option(SOFAMISCFORCEFIELD_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCFORCEFIELD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscForceField_test)
endif()