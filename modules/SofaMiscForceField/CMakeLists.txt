include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initMiscForcefield.h
    MatrixMass.h
    MatrixMass.inl
    MeshMatrixMass.h
    MeshMatrixMass.inl
    LennardJonesForceField.h
    LennardJonesForceField.inl
    WashingMachineForceField.h
    WashingMachineForceField.inl
    GearSpringForceField.h
    GearSpringForceField.inl
    LineBendingSprings.h
    LineBendingSprings.inl
    )

set(SOURCE_FILES

    initMiscForcefield.cpp
    MatrixMass.cpp
    MeshMatrixMass.cpp
    LennardJonesForceField.cpp
    WashingMachineForceField.cpp
    GearSpringForceField.cpp
    LineBendingSprings.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_FORCEFIELD" )
AddLinkerDependencies(SofaDeformable SofaBoundaryCondition SofaMiscTopology )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
