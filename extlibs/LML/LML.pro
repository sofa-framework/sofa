#
# load.pro
# 

SOFA_DIR = ../..
TARGET = load

include($${SOFA_DIR}/sofa.cfg)

TEMPLATE = lib
CONFIG += $$CONFIGLIBRARIES
LIBS *= $$SOFA_EXT_LIBS
LIBS -= -lload$$LIBSUFFIX

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
	   LoadsVersion.h \
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



