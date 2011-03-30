# Target is a library: sofagui
load(sofa-pre)

TEMPLATE = lib
TARGET = sofaguimain

CONFIG += qt
QT += qt3support

INCLUDEPATH += $$BUILD_DIR/qt # HACK: GUI.h is generated in other .pro, this will not work if UI_DIR is changed
DEFINES += SOFA_BUILD_GUIMANAGER

SOURCES = \
	GUIManager.cpp

HEADERS = \ 
	GUIManager.h

load(sofa-post)
