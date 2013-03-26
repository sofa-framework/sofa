cmake_minimum_required(VERSION 2.8)

project("SofaMiscForceField")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initMiscForcefield.h 
    mass/MatrixMass.h 
    mass/MatrixMass.inl 
    mass/MeshMatrixMass.h 
    mass/MeshMatrixMass.inl 
    forcefield/LennardJonesForceField.h 
    forcefield/LennardJonesForceField.inl 
    forcefield/WashingMachineForceField.h 
    forcefield/WashingMachineForceField.inl 
    interactionforcefield/GearSpringForceField.h 
    interactionforcefield/GearSpringForceField.inl 
    interactionforcefield/LineBendingSprings.h 
    interactionforcefield/LineBendingSprings.inl

    )
    
set(SOURCE_FILES

    initMiscForcefield.cpp 
    mass/MatrixMass.cpp 
    mass/MeshMatrixMass.cpp 
    forcefield/LennardJonesForceField.cpp 
    forcefield/WashingMachineForceField.cpp 
    interactionforcefield/GearSpringForceField.cpp 
    interactionforcefield/LineBendingSprings.cpp
 
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_MISC_FORCEFIELD" )
set(LINKER_DEPENDENCIES SofaDeformable SofaBoundaryCondition SofaMiscTopology )
    
include(${SOFA_CMAKE_DIR}/post.cmake)
