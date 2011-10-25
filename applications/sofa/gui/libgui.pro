# Target is a library: sofagui
load(sofa/pre)

TEMPLATE = lib
TARGET = sofagui

INCLUDEPATH += $$ROOT_SRC_DIR/applications
DEPENDPATH += $$ROOT_SRC_DIR/applications
DEFINES += SOFA_BUILD_SOFAGUI

SOURCES = \
	BatchGUI.cpp \
	ColourPickingVisitor.cpp \
	FilesRecentlyOpenedManager.cpp \
	MouseOperations.cpp \
	PickHandler.cpp \
	SofaGUI.cpp

HEADERS = \
	BatchGUI.h \
	ColourPickingVisitor.h \
	MouseOperations.h \
	OperationFactory.h \
	PickHandler.h \
	FilesRecentlyOpenedManager.h \
	SofaGUI.h

load(sofa/post)
