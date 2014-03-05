include(${SOFA_CMAKE_DIR}/preProject.cmake)

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


if(SOFA_GPU_CUDA)
    
#    list(APPEND HEADER_FILES "HexaRemover.h")
#    list(APPEND HEADER_FILES "HexaRemover.inl")
#    list(APPEND SOURCE_FILES "HexaRemover.cpp")

endif()

set(README_FILES

	frame.txt
	)	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_FRAME")
AddLinkerDependencies(SofaGuiQt SofaSparseSolver SofaEigen2Solver)

if(UNIX)
    find_package(X11)
    if(NOT X11_FOUND)
        message(FATAL_ERROR "Failed to find X11 which is required to build frame plugin")
    endif(NOT X11_FOUND)
    AddLinkerDependencies(${X11_X11_LIB})
    include_directories(${X11_X11_INCLUDE_PATH})
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
