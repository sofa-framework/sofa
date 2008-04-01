SOFA_DIR=../../..
TEMPLATE = app

include($${SOFA_DIR}/sofa.cfg)

TARGET = sofaCUDA$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTGUI
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

SOURCES = Main.cpp \
	  ComputedCoil.cpp
HEADERS = ComputedCoil.h

RC_FILE = sofa.rc
