cmake_minimum_required(VERSION 2.8)

project("SofaFrame")

include(${SOFA_CMAKE_DIR}/pre.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    initFrame.h 
    AffineTypes.h 
    DualQuatBlending.h 
    QuadraticTypes.h 
    CorotationalForceField.h 
    CorotationalForceField.inl 
    GreenLagrangeForceField.h 
    GreenLagrangeForceField.inl 
    FrameVolumePreservationForceField.h 
    FrameVolumePreservationForceField.inl 
    DeformationGradientTypes.h 
    Blending.h 
    LinearBlending.h 
    LinearBlending.inl 
    DualQuatBlending.h 
    DualQuatBlending.inl 
    FrameBlendingMapping.h 
    FrameBlendingMapping.inl 
    FrameConstantForceField.h 
    FrameMass.h 
    FrameDiagonalMass.h 
    FrameDiagonalMass.inl 
    FrameFixedConstraint.h 
    FrameFixedConstraint.inl 
    FrameLinearMovementConstraint.h 
    FrameRigidConstraint.h 
    FrameRigidConstraint.inl 
    FrameMechanicalObject.h 
    GridMaterial.h 
    GridMaterial.inl 
    MeshGenerator.h 
    MeshGenerator.inl 
    NewMaterial.h 
    NewHookeMaterial.h 
    NewHookeMaterial.inl
    #TetrahedronFEMForceFieldWithExternalMaterial.h 
    #TetrahedronFEMForceFieldWithExternalMaterial.inl

	)

set(SOURCE_FILES

    initFrame.cpp 
    CorotationalForceField.cpp 
    GreenLagrangeForceField.cpp 
    FrameVolumePreservationForceField.cpp 
    FrameBlendingMapping.cpp 
    FrameBlendingMapping-Rigid-2.cpp 
    FrameBlendingMapping-Affine3d.cpp 
    FrameBlendingMapping-Affine3d-2.cpp 
    FrameBlendingMapping-Quadratic3d.cpp 
    FrameBlendingMapping-Quadratic3d-2.cpp 
    FrameConstantForceField.cpp 
    FrameDiagonalMass.cpp 
    FrameFixedConstraint.cpp 
    FrameLinearMovementConstraint.cpp 
    FrameRigidConstraint.cpp 
    FrameMechanicalObject.cpp 
    GridMaterial.cpp 
    MeshGenerator.cpp 
    NewHookeMaterial.cpp
    #TetrahedronFEMForceFieldWithExternalMaterial.cpp

	)

set(README_FILE frame.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_FRAME")
set(LINKER_DEPENDENCIES SofaGuiQt SofaSparseSolver SofaEigen2Solver)

include(${SOFA_CMAKE_DIR}/post.cmake)