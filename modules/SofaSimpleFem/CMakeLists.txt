cmake_minimum_required(VERSION 2.8.12)
project(SofaSimpleFem)

set(HEADER_FILES
    BeamFEMForceField.h
    BeamFEMForceField.inl
    HexahedralFEMForceField.h
    HexahedralFEMForceField.inl
    HexahedralFEMForceFieldAndMass.h
    HexahedralFEMForceFieldAndMass.inl
    HexahedronFEMForceField.h
    HexahedronFEMForceField.inl
    HexahedronFEMForceFieldAndMass.h
    HexahedronFEMForceFieldAndMass.inl
    LengthContainer.h
    PoissonContainer.h
    RadiusContainer.h
    StiffnessContainer.h
    TetrahedralCorotationalFEMForceField.h
    TetrahedralCorotationalFEMForceField.inl
    TetrahedronFEMForceField.h
    TetrahedronFEMForceField.inl
    TetrahedronDiffusionFEMForceField.h
    TetrahedronDiffusionFEMForceField.inl
    TriangularFEMForceFieldOptim.h
    TriangularFEMForceFieldOptim.inl
    config.h
    initSimpleFEM.h
)

set(SOURCE_FILES
    initSimpleFEM.cpp
    BeamFEMForceField.cpp
    HexahedralFEMForceField.cpp
    HexahedralFEMForceFieldAndMass.cpp
    HexahedronFEMForceField.cpp
    HexahedronFEMForceFieldAndMass.cpp
    TetrahedralCorotationalFEMForceField.cpp
    TetrahedronFEMForceField.cpp
    TetrahedronDiffusionFEMForceField.cpp
    TriangularFEMForceFieldOptim.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIMPLE_FEM")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
