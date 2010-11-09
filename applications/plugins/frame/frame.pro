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
           FrameConstantForceField.h \
           FrameDiagonalMass.h \
           FrameDiagonalMass.inl \
           FrameFixedConstraint.h \
           FrameForceField.h \
           FrameHookeForceField.h \
           FrameHookeForceField.inl \
           FrameLoydAlgo.h \
           FrameLoydAlgo.inl \
           FrameMass.h \
           FrameMechanicalObject.h \
           FrameSpringForceField2.h \
           FrameSpringForceField2.inl \
           initFrame.h \
           QuadraticTypes.h

SOURCES += \
           FrameConstantForceField.cpp \
           FrameDiagonalMass.cpp \
           FrameFixedConstraint.cpp \
           FrameForceField.cpp \
           FrameHookeForceField.cpp \
           FrameLoydAlgo.cpp \
           FrameMechanicalObject.cpp \
           FrameSpringForceField2.cpp \
           initFrame.cpp
