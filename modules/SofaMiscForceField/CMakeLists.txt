cmake_minimum_required(VERSION 3.1)
project(SofaMiscForceField)

set(HEADER_FILES
    ForceMaskOff.h
    GearSpringForceField.h
    GearSpringForceField.inl
    LennardJonesForceField.h
    LennardJonesForceField.inl
    LineBendingSprings.h
    LineBendingSprings.inl
    MatrixMass.h
    MatrixMass.inl
    MeshMatrixMass.h
    MeshMatrixMass.inl
    WashingMachineForceField.h
    WashingMachineForceField.inl
    config.h
    initMiscForcefield.h
)

set(SOURCE_FILES
    ForceMaskOff.cpp
    GearSpringForceField.cpp
    LennardJonesForceField.cpp
    LineBendingSprings.cpp
    MatrixMass.cpp
    MeshMatrixMass.cpp
    WashingMachineForceField.cpp
    initMiscForcefield.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDeformable SofaBoundaryCondition SofaMiscTopology SofaGeneralTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_FORCEFIELD")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
