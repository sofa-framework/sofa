cmake_minimum_required(VERSION 3.12)
project(SofaMiscForceField VERSION 1.0 LANGUAGES CXX)

find_package(SofaFramework REQUIRED) # SofaHelper
find_package(SofaCommon REQUIRED) # SofaDeformable
find_package(SofaGeneral REQUIRED) # SofaBoundaryCondition SofaGeneralTopology

set(SOFAMISCFORCEFIELD_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCFORCEFIELD_SRC}/config.h.in
    ${SOFAMISCFORCEFIELD_SRC}/initMiscForcefield.h
    )

set(SOURCE_FILES
    ${SOFAMISCFORCEFIELD_SRC}/initMiscForcefield.cpp
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

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDeformable SofaBoundaryCondition SofaMiscTopology SofaGeneralTopology)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )
