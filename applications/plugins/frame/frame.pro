SOFA_DIR =../../..
TEMPLATE = lib
TARGET = sofaframe
include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

CONFIG += $$CONFIGLIBRARIES

!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

DEFINES += SOFA_BUILD_FRAME

LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += $$SOFA_EXT_LIBS
LIBS += -lsofasimulation$$LIBSUFFIX
LIBS += -lsofacomponentprojectiveconstraintset$$LIBSUFFIX
LIBS += -lsofacomponentbase$$LIBSUFFIX

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_DIR/applications

HEADERS += \
           AffineTypes.h \
           CorotationalForceField.h \
           CorotationalForceField.inl \
           DeformationGradientTypes.h \
           FrameBlendingMapping.h \
           FrameBlendingMapping.inl \
           FrameConstantForceField.h \
           FrameDiagonalMass.h \
           FrameDiagonalMass.inl \
           FrameDQBlendingMapping.h \
           FrameDQBlendingMapping.inl \
           FrameFixedConstraint.h \
           FrameFixedConstraint.inl \
           FrameForceField.h \
           FrameForceField.inl \
           FrameLinearMovementConstraint.h \
           FrameMass.h \
           FrameMechanicalObject.h \
           FrameRigidConstraint.h \
           FrameRigidConstraint.inl \
           FrameVolumePreservationForceField.h \
           FrameVolumePreservationForceField.inl \
           GridMaterial.h \
           GridMaterial.inl \
           Interpolater.h \
           Interpolater.inl \
           initFrame.h \
           MappingTypes.h \
           NewMaterial.h \
           NewHookeMaterial.h \
           NewHookeMaterial.inl \
           QuadraticTypes.h \
           TetrahedralCorotationalFEMForceField2.h \
           TetrahedralCorotationalFEMForceField2.inl

SOURCES += \
           CorotationalForceField.cpp \
           FrameBlendingMapping.cpp \
           FrameConstantForceField.cpp \
           FrameDiagonalMass.cpp \
           FrameDQBlendingMapping.cpp \
           FrameFixedConstraint.cpp \
           FrameForceField.cpp \
           FrameLinearMovementConstraint.cpp \
           FrameMechanicalObject.cpp \
           FrameRigidConstraint.cpp \
           FrameVolumePreservationForceField.cpp \
           GridMaterial.cpp \
           Interpolater.cpp \
           NewHookeMaterial.cpp \
           TetrahedralCorotationalFEMForceField2.cpp \
           initFrame.cpp
