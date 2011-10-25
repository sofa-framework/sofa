######  PLUGIN TARGET
TARGET = MeshSTEPLoaderPlugin

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
CONFIG += $$CONFIGLIBRARIES
CONFIG -= staticlib
CONFIG += dll

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin

TARGET = MeshSTEPLoaderPlugin$$LIBSUFFIX
DEFINES += SOFA_BUILD_MESHSTEPLOADERPLUGIN

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS


INCLUDEPATH += $$SOFA_DIR/extlibs
DEPENDPATH += $$SOFA_DIR/extlibs


SOURCES = \
MeshSTEPLoader.cpp \
SingleComponent.cpp \
          initMeshSTEPLoader.cpp

HEADERS = \
MeshSTEPLoader.h\
SingleComponent.inl\
SingleComponent.h



README_FILE = PluginMeshSTEPLoader.txt

unix {
    INCLUDEPATH += /usr/include/opencascade
DEPENDPATH += /usr/include/opencascade
    LIBS += -lTKernel -lTKMath -lTKAdvTools -lGL -lTKG2d -lTKG3d -lTKGeomBase -lTKBRep -lTKGeomAlgo -lTKTopAlgo -lTKPrim -lTKBO -lTKHLR -lTKMesh -lTKShHealing -lTKBool -lTKXMesh -lTKFillet -lTKFeat -lTKOffset -lTKSTL -lTKXSBase -lTKSTEPBase -lTKIGES -lTKSTEPAttr -lTKSTEP209 -lTKSTEP    -lTKService -lTKV2d -lTKV3d -lTKOpenGl -lTKMeshVS -lTKNIS -lTKVRML
    QMAKE_POST_LINK = cp $$README_FILE $$DESTDIR
}

win32 {
    INCLUDEPATH += $$OPEN_CASCADE_DIR/inc
DEPENDPATH += $$OPEN_CASCADE_DIR/inc
    LIBS += -l$$OPEN_CASCADE_DIR/win32/lib/*
    QMAKE_CXXFLAGS += /DWNT
    QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"
}
