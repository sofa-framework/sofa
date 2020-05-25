cmake_minimum_required(VERSION 3.1)
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

sofa_install_targets(SofaCommon ${PROJECT_NAME} "SofaCommon/${PROJECT_NAME}")
