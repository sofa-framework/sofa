SOFA_DIR =../../..
TEMPLATE = lib
TARGET = sofaframe
include($${SOFA_DIR}/sofa.cfg)

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
           FrameDiagonalMass.h \
           FrameDiagonalMass.inl \
           FrameFixedConstraint.h \
           FrameForcefield.h \
           FrameHookeForceField.h \
           FrameHookeForceField.inl \
           FrameMass.h \
           FrameMechanicalObject.h \
           FrameSpringForceField2.h \
           FrameSpringForceField2.inl \
           QuadratiqueTypes.h

SOURCES += \
           FrameDiagonalMass.cpp \
           FrameFixedConstraint.cpp \
           FrameForcefield.cpp \
           FrameHookeForceField.cpp \
           FrameMechanicalObject.cpp \
           FrameSpringForceField2.cpp
