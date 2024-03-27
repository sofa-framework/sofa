cmake_minimum_required(VERSION 3.22)
project(SofaMiscForceField LANGUAGES CXX)

set(SOFAMISCFORCEFIELD_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCFORCEFIELD_SRC}/config.h.in
    ${SOFAMISCFORCEFIELD_SRC}/initSofaMiscForceField.h
    )

set(SOURCE_FILES
    ${SOFAMISCFORCEFIELD_SRC}/initSofaMiscForceField.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Mass and Sofa.Component.SolidMechanics.Spring instead.")

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
    RELOCATABLE "collections"
    )
