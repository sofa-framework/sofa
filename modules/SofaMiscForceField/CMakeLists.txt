cmake_minimum_required(VERSION 3.1)
project(SofaMiscForceField VERSION 1.0 LANGUAGES CXX)

find_package(SofaBase REQUIRED)
find_package(SofaGeneral REQUIRED)
find_package(SofaMiscTopology REQUIRED)

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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_FORCEFIELD")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")


target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src/${PROJECT_NAME}"
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

