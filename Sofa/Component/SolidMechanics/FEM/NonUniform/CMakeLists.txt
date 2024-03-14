cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.SolidMechanics.FEM.NonUniform LANGUAGES CXX)

set(SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR "src/sofa/component/solidmechanics/fem/nonuniform/")

set(HEADER_FILES
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/init.h
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/HexahedronCompositeFEMForceFieldAndMass.h
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/HexahedronCompositeFEMForceFieldAndMass.inl
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/HexahedronCompositeFEMMapping.h
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/HexahedronCompositeFEMMapping.inl
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/NonUniformHexahedralFEMForceFieldAndMass.h
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/NonUniformHexahedralFEMForceFieldAndMass.inl
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/NonUniformHexahedronFEMForceFieldAndMass.h
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/NonUniformHexahedronFEMForceFieldAndMass.inl
)

set(SOURCE_FILES
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/HexahedronCompositeFEMForceFieldAndMass.cpp
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/HexahedronCompositeFEMMapping.cpp
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/NonUniformHexahedralFEMForceFieldAndMass.cpp
    ${SOFACOMPONENTSOLIDMECHANICSFEMNONUNIFORM_SOURCE_DIR}/NonUniformHexahedronFEMForceFieldAndMass.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Grid REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.FEM.Elastic)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Grid)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Dynamic)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
