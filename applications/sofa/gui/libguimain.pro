# Target is a library: sofagui
load(sofa/pre)

TEMPLATE = lib
TARGET = sofaguimain

CONFIG += qt
QT += qt3support

INCLUDEPATH += $$BUILD_DIR/qt/$$UI_DIR # HACK: GUI.h is generated in other .pro
DEPENDPATH += $$BUILD_DIR/qt/$$UI_DIR # HACK: GUI.h is generated in other .pro
DEFINES += SOFA_BUILD_GUIMAIN

SOURCES = \
	Main.cpp

HEADERS = \
	Main.h

load(sofa/post)
