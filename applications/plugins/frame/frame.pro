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
 #          AffineSkinningMapping.h \
 #          AffineSkinningMapping.inl \
           DeformationGradientTypes.h \
           FrameBlendingMapping.h \
           FrameBlendingMapping.inl \
           FrameConstantForceField.h \
           FrameDiagonalMass.h \
           FrameDiagonalMass.inl \
 #           FrameDualQuatSkinningMapping.h \
 #           FrameDualQuatSkinningMapping.inl \
           FrameFixedConstraint.h \
           FrameForceField.h \
#           FrameHookeForceField.h \
 #          FrameHookeForceField.inl \
           FrameLoydAlgo.h \
           FrameLoydAlgo.inl \
           FrameMass.h \
           FrameMechanicalObject.h \
           FrameSampler.h \
           FrameSampler.inl \
#           FrameSpringForceField2.h \
#           FrameSpringForceField2.inl \
           GridMaterial.h \
           GridMaterial.inl \
           initFrame.h \
           MappingTypes.h \
            NewMaterial.h \
            NewHookeMaterial.h \
            NewHookeMaterial.inl \
#           PrimitiveSkinningMapping.h \
#           PrimitiveSkinningMapping.inl \
#           RigidSkinningMapping.h \
#           RigidSkinningMapping.inl \
#           QuadraticSkinningMapping.h \
#           QuadraticSkinningMapping.inl \
           QuadraticTypes.h
           
SOURCES += \
#           AffineSkinningMapping.cpp \
           FrameBlendingMapping.cpp \
           FrameConstantForceField.cpp \
           FrameDiagonalMass.cpp \
 #           FrameDualQuatSkinningMapping.cpp \
           FrameFixedConstraint.cpp \
           FrameForceField.cpp \
#           FrameHookeForceField.cpp \
           FrameLoydAlgo.cpp \
           FrameMechanicalObject.cpp \
           FrameSampler.cpp \
 #          FrameSpringForceField2.cpp \
           GridMaterial.cpp \
            NewHookeMaterial.cpp \
#           PrimitiveSkinningMapping.cpp \
#           QuadraticSkinningMapping.cpp \
#           RigidSkinningMapping.cpp \
           initFrame.cpp
