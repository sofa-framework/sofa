cmake_minimum_required(VERSION 3.12)
project(SofaSimpleFem)

set(HEADER_FILES
    HexahedronFEMForceField.h
    HexahedronFEMForceField.inl
    TetrahedronFEMForceField.h
    TetrahedronFEMForceField.inl
    TetrahedronDiffusionFEMForceField.h
    TetrahedronDiffusionFEMForceField.inl
    config.h
    initSimpleFEM.h
)

set(SOURCE_FILES
    initSimpleFEM.cpp
    HexahedronFEMForceField.cpp
    TetrahedronFEMForceField.cpp
    TetrahedronDiffusionFEMForceField.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaCommon
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
