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

sofa_find_package(SofaFramework REQUIRED) # SofaHelper
sofa_find_package(SofaBaseMechanics REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)

message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. You may use Sofa.Component.Mass and Sofa.Component.SolidMechanics.Spring instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mass REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mass)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.FEM.Elastic)

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
