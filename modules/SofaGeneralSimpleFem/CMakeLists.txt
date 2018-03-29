cmake_minimum_required(VERSION 3.1)
project(SofaGeneralSimpleFem)

set(HEADER_FILES
    config.h
    initGeneralSimpleFEM.h
    )

set(SOURCE_FILES
    initGeneralSimpleFEM.cpp
    )

list(APPEND HEADER_FILES
    BeamFEMForceField.h
    BeamFEMForceField.inl
    HexahedralFEMForceField.h
    HexahedralFEMForceField.inl
    HexahedralFEMForceFieldAndMass.h
    HexahedralFEMForceFieldAndMass.inl
    HexahedronFEMForceFieldAndMass.h
    HexahedronFEMForceFieldAndMass.inl
    LengthContainer.h
    PoissonContainer.h
    RadiusContainer.h
    StiffnessContainer.h
    TetrahedralCorotationalFEMForceField.h
    TetrahedralCorotationalFEMForceField.inl
    TriangularFEMForceFieldOptim.h
    TriangularFEMForceFieldOptim.inl
    )
list(APPEND SOURCE_FILES
    BeamFEMForceField.cpp
    HexahedralFEMForceField.cpp
    HexahedralFEMForceFieldAndMass.cpp
    HexahedronFEMForceFieldAndMass.cpp
    TetrahedralCorotationalFEMForceField.cpp
    TriangularFEMForceFieldOptim.cpp

    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaSimpleFem)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_SIMPLE_FEM")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
