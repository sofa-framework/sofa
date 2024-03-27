cmake_minimum_required(VERSION 3.22)
project(SofaNonUniformFem LANGUAGES CXX)

set(SOFANONUNIFORMFEM_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFANONUNIFORMFEM_SRC}/config.h.in
    ${SOFANONUNIFORMFEM_SRC}/initSofaNonUniformFem.h
    )

set(SOURCE_FILES
    ${SOFANONUNIFORMFEM_SRC}/initSofaNonUniformFem.cpp
    )

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Topology.Container.Grid, Sofa.Component.Topology.Container.Dynamic and Sofa.Component.SolidMechanics.FEM.NonUniform instead.")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

# forward to the new sofang module
sofa_find_package(Sofa.Component.Topology.Container.Grid REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.NonUniform REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Grid)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Dynamic)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.FEM.NonUniform)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
