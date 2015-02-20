include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMiscForcefield.h 
    # ../mass/MatrixMass.h 
    # ../mass/MatrixMass.inl 
    # ../mass/MeshMatrixMass.h 
    # ../mass/MeshMatrixMass.inl 
    # ../forcefield/LennardJonesForceField.h 
    # ../forcefield/LennardJonesForceField.inl 
    # ../forcefield/WashingMachineForceField.h 
    # ../forcefield/WashingMachineForceField.inl 
#     ../interactionforcefield/GearSpringForceField.h
#     ../interactionforcefield/GearSpringForceField.inl
    # ../interactionforcefield/LineBendingSprings.h 
    # ../interactionforcefield/LineBendingSprings.inl

    )
    
set(SOURCE_FILES

    ../initMiscForcefield.cpp 
    # ../mass/MatrixMass.cpp 
    # ../mass/MeshMatrixMass.cpp 
    # ../forcefield/LennardJonesForceField.cpp 
    # ../forcefield/WashingMachineForceField.cpp 
#     ../interactionforcefield/GearSpringForceField.cpp
    # ../interactionforcefield/LineBendingSprings.cpp
 
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_FORCEFIELD" )
AddLinkerDependencies(SofaDeformable SofaBoundaryCondition SofaMiscTopology )
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
