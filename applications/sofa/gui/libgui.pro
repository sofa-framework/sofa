# Target is a library: sofagui
load(sofa-pre)

TEMPLATE = lib
TARGET = sofagui

INCLUDEPATH += $$ROOT_SRC_DIR/applications
DEFINES += SOFA_BUILD_SOFAGUI

SOURCES = \
	SofaGUI.cpp \
	BatchGUI.cpp \
	MouseOperations.cpp \
	PickHandler.cpp \
	FilesRecentlyOpenedManager.cpp

HEADERS = \
	SofaGUI.h \
	BatchGUI.h \
	OperationFactory.h \
	MouseOperations.h \
	PickHandler.h \
	FilesRecentlyOpenedManager.h 

load(sofa-post)
