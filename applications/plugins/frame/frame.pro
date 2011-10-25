load(sofa/pre)
defineAsPlugin(frame)

TEMPLATE = lib
TARGET = sofaframe

DEFINES += SOFA_BUILD_FRAME

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

HEADERS += \
           initFrame.h \
           AffineTypes.h \
           DualQuatTypes.h \
           QuadraticTypes.h \
           CorotationalForceField.h \
           CorotationalForceField.inl \
           GreenLagrangeForceField.h \
           GreenLagrangeForceField.inl \
           FrameVolumePreservationForceField.h \
           FrameVolumePreservationForceField.inl \
           DeformationGradientTypes.h \
           MappingTypes.h \
           FrameBlendingMapping.h \
           FrameBlendingMapping.inl \
           LinearBlendTypes.inl \
           DualQuatBlendTypes.inl \
           FrameConstantForceField.h \
           FrameMass.h \
           FrameDiagonalMass.h \
           FrameDiagonalMass.inl \
           FrameFixedConstraint.h \
           FrameFixedConstraint.inl \
           FrameLinearMovementConstraint.h \
           FrameRigidConstraint.h \
           FrameRigidConstraint.inl \
           FrameMechanicalObject.h \
           GridMaterial.h \
           GridMaterial.inl \
           MeshGenerater.h \
           MeshGenerater.inl \
           NewMaterial.h \
           NewHookeMaterial.h \
           NewHookeMaterial.inl
           #TetrahedronFEMForceFieldWithExternalMaterial.h \
           #TetrahedronFEMForceFieldWithExternalMaterial.inl
SOURCES += \
           initFrame.cpp \
           CorotationalForceField.cpp \
           GreenLagrangeForceField.cpp \
           FrameVolumePreservationForceField.cpp \
           FrameBlendingMapping.cpp \
           FrameConstantForceField.cpp \
           FrameDiagonalMass.cpp \
           FrameFixedConstraint.cpp \
           FrameLinearMovementConstraint.cpp \
           FrameRigidConstraint.cpp \
           FrameMechanicalObject.cpp \
           GridMaterial.cpp \
           MeshGenerater.cpp \
           NewHookeMaterial.cpp
           #TetrahedronFEMForceFieldWithExternalMaterial.cpp

contains(DEFINES, SOFA_GPU_CUDA) { # BEGIN SOFA_GPU_CUDA

HEADERS += \
           HexaRemover.h \
           HexaRemover.inl

SOURCES += \
           HexaRemover.cpp
}

load(sofa/post)
