cmake_minimum_required(VERSION 2.8.12)
project(SofaMiscForceField)

set(HEADER_FILES
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
    GearSpringForceField.cpp
    LennardJonesForceField.cpp
    LineBendingSprings.cpp
    MatrixMass.cpp
    MeshMatrixMass.cpp
    WashingMachineForceField.cpp
    initMiscForcefield.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaDeformable SofaBoundaryCondition SofaMiscTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_FORCEFIELD -DPOINT_DATA_VECTOR_ACCESS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaMisc_install(${PROJECT_NAME})
