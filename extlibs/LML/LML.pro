#
# load.pro
# 

SOFA_DIR = ../..
include($$SOFA_DIR/sofa.cfg)

TARGET = load$$LIBSUFFIX
TEMPLATE = lib
CONFIG += $$CONFIGLIBRARIES


# --------------- Sources --------------------
SOURCES += Force.cpp \
           ForceUnit.cpp \
           Load.cpp \
           Loads.cpp \
           PressureUnit.cpp \
           RotationUnit.cpp \
           TargetList.cpp \
           Translation.cpp \
           TranslationUnit.cpp \
           ValueEvent.cpp \
	     XMLLoads.cpp \



HEADERS += Direction.h \
           Force.h \
           ForceUnit.h \
           Load.h \
           Loads.h \
           Pressure.h \
           PressureUnit.h \
           Rotation.h \
           RotationUnit.h \
           TargetList.h \
           Translation.h \
           TranslationUnit.h \
           Unit.h \
           ValueEvent.h \
	     XMLLoads.h \
		xmlio.h



