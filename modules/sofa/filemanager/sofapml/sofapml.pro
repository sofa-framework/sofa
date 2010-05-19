SOFA_DIR = ../../../..
TEMPLATE = lib
TARGET = sofapml

include($${SOFA_DIR}/sofa.cfg)


CONFIG += $$CONFIGLIBRARIES

!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

DEFINES += SOFA_BUILD_FILEMANAGER_PML

LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += $$SOFA_EXT_LIBS
LIBS += $$SOFA_MODULES_LIBS
LIBS -= -lsofapml$$LIBSUFFIX

HEADERS += sofapml.h \
           PMLBody.h \
           PMLRigidBody.h \
           PMLFemForceField.h \
           PMLStiffSpringForceField.h \
           PMLInteractionForceField.h \
           PMLMappedBody.h \
           PMLReader.h \
           LMLConstraint.h \
           LMLConstraint.inl \
           LMLForce.h \
           LMLForce.inl \
           LMLReader.h


SOURCES += PMLBody.cpp \
           PMLRigidBody.cpp \
           PMLFemForceField.cpp \
           PMLStiffSpringForceField.cpp \
           PMLInteractionForceField.cpp \
           PMLMappedBody.cpp \
           PMLReader.cpp \
           LMLConstraint.cpp \
           LMLForce.cpp \
           LMLReader.cpp 

