cmake_minimum_required(VERSION 3.12)
project(SofaGeneralDeformable LANGUAGES CXX)

set(SOFAGENERALDEFORMABLE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALDEFORMABLE_SRC}/config.h.in
    ${SOFAGENERALDEFORMABLE_SRC}/initSofaGeneralDeformable.h
    )

set(SOURCE_FILES
    ${SOFAGENERALDEFORMABLE_SRC}/initSofaGeneralDeformable.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.SolidMechanics.Spring and Sofa.Component.SolidMechanics.TensorMass instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.SolidMechanics.Spring REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.TensorMass REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.Spring)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.TensorMass)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
