cmake_minimum_required(VERSION 3.1)
project(SofaMiscForceField)

set(SOFAMISCFORCEFIELD_MAJOR_VERSION 0)
set(SOFAMISCFORCEFIELD_MINOR_VERSION 1)
set(SOFAMISCFORCEFIELD_VERSION ${SOFAMISCFORCEFIELD_MAJOR_VERSION}.${SOFAMISCFORCEFIELD_MINOR_VERSION})

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

## Install rules for the library; CMake package configurations files
sofa_create_package(SofaMiscForceField ${SOFAMISCFORCEFIELD_VERSION} SofaMiscForceField SofaMiscForceField)
