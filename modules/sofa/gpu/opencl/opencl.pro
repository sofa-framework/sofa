# Target is a library: sofagpuopencl
SOFA_DIR = ../../../..
TEMPLATE = lib
TARGET = sofagpuopencl
DEFINES += SOFA_BUILD_GPU_OPENCL
include($${SOFA_DIR}/sofa.cfg)
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll
LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += -lsofasimulation$$LIBSUFFIX
LIBS += -lsofatree$$LIBSUFFIX
LIBS += -lsofaautomatescheduler$$LIBSUFFIX
LIBS += -lsofacomponentbase$$LIBSUFFIX
LIBS += -lsofacomponentmastersolver$$LIBSUFFIX
LIBS += -lsofacomponentfem$$LIBSUFFIX
LIBS += -lsofacomponentinteractionforcefield$$LIBSUFFIX
LIBS += -lsofacomponentcontextobject$$LIBSUFFIX
LIBS += -lsofacomponentbehaviormodel$$LIBSUFFIX
LIBS += -lsofacomponentlinearsolver$$LIBSUFFIX
LIBS += -lsofacomponentodesolver$$LIBSUFFIX
LIBS += -lsofacomponentbase$$LIBSUFFIX
LIBS += -lsofacomponentcontroller$$LIBSUFFIX
LIBS += -lsofacomponentengine$$LIBSUFFIX
LIBS += -lsofacomponentvisualmodel$$LIBSUFFIX
LIBS += -lsofacomponentmass$$LIBSUFFIX
LIBS += -lsofacomponentforcefield$$LIBSUFFIX
LIBS += -lsofacomponentmapping$$LIBSUFFIX
LIBS += -lsofacomponentconstraint$$LIBSUFFIX
LIBS += -lsofacomponentcollision$$LIBSUFFIX
LIBS += -lsofacomponentmisc$$LIBSUFFIX
LIBS += -lsofacomponent$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS


include(0Core.pro)
include(0MechanicalObject.pro)
include(0MemoryManager.pro)
include(tools/tools.pro)

HEADERS += OpenCLFixedConstraint.h \
    OpenCLFixedConstraint.inl \
    OpenCLIdentityMapping.h \
    OpenCLIdentityMapping.inl \
    OpenCLPlaneForceField.h \
    OpenCLPlaneForceField.inl \
    OpenCLSphereForceField.h \
    OpenCLSphereForceField.inl \
    OpenCLUniformMass.h \
    OpenCLUniformMass.inl \
    OpenCLSpringForceField.h \
	OpenCLSpringForceField.inl

SOURCES += OpenCLBoxROI.cpp \
    OpenCLFixedConstraint.cpp \
    OpenCLIdentityMapping.cpp \
    OpenCLSpringForceField.cpp \
    OpenCLPlaneForceField.cpp \
    OpenCLSphereForceField.cpp \
	OpenCLUniformMass.cpp

include(kernels/kernels.pro)




