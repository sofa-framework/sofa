SOFA_DIR = ../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofapml$$LIBSUFFIX

CONFIG += $$CONFIGLIBRARIES
LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += $$SOFA_EXT_LIBS

HEADERS += PMLBody.h \
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


SOURCES = 	PMLBody.cpp \
           PMLRigidBody.cpp \
           PMLFemForceField.cpp \
           PMLStiffSpringForceField.cpp \
	     PMLInteractionForceField.cpp \
           PMLMappedBody.cpp \
           PMLReader.cpp \
           LMLConstraint.cpp \
           LMLForce.cpp \
           LMLReader.cpp 

