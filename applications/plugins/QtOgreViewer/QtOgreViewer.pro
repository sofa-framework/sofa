######  PLUGIN TARGET
TARGET = qtogreviewerplugin


###### PREREQUISITES
# SOFA_GUI_QT
# SOFA_QT4
######

######  GENERAL PLUGIN CONFIGURATION, you shouldn't have to modify it

SOFA_DIR=../../..
TEMPLATE = lib

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/lib/sofa-plugins

#set configuration to dynamic library
contains (DEFINES, SOFA_QT4) {
CONFIG += $$CONFIGLIBRARIES qt
QT += opengl qt3support xml
}
else{
CONFIG += $$CONFIGLIBRARIES qt
QT += opengl
}
CONFIG -= staticlib
CONFIG += dll

###### SPECIFIC PLUGIN CONFIGURATION, you should modify it to configure your plugin


DEFINES += SOFA_BUILD_QTOGREVIEWERPLUGIN
DEFINES += SOFA_GUI_QT

LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS
LIBS += $$SOFA_GUI_LIBS

INCLUDEPATH += $$SOFA_DIR/extlibs


SOURCES += DotSceneLoader.cpp \
HelperLogics.cpp \
QtOgreViewer.cpp\
QtOgreViewer_slots.cpp\
OgreVisualModel.cpp \
OgreMeshLoader.cpp \
OgreShaderParameter.cpp \
OgreShaderTextureUnit.cpp \
QOgreLightWidget.cpp \
OgrePlanarReflectionMaterial.cpp \
DrawToolOGRE.cpp \
OgreViewerSetting.cpp \
OgreSceneObject.cpp \
PropagateOgreSceneManager.cpp \
initQtOgreViewer.cpp \
SubMesh.cpp 

HEADERS += DotSceneLoader.h \
HelperLogics.h \
QtOgreViewer.h \
OgreVisualModel.h \
OgreMeshLoader.h \
OgreShaderEntryPoint.h \
OgreShaderParameter.h \
OgreShaderTextureUnit.h \
QOgreLightWidget.h \
OgrePlanarReflectionMaterial.h \
OgreSofaViewer.h \
DrawToolOGRE.h \
OgreViewerSetting.h \
OgreSceneObject.h \
PropagateOgreSceneManager.h \
initQtOgreViewer.h \
SubMesh.h


README_FILE = QtOgreViewer.txt

macx {
QMAKE_CXXFLAGS += -Wno-unused
LIBS += -framework Ogre -framework CoreFoundation
}

win32 {

INCLUDEPATH += $${OGRE_HOME}/include/OGRE/

contains (CONFIGDEBUG, debug) {
QMAKE_LIBDIR += $${OGRE_HOME}/lib/debug/
LIBS += OgreMain_d.lib
}
else{
QMAKE_LIBDIR += $${OGRE_HOME}/lib/release/
LIBS += OgreMain.lib
}
QMAKE_POST_LINK = copy \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"
}


unix {
!macx: {
QMAKE_CXXFLAGS += $$system(pkg-config --cflags OGRE )
LIBS += $$system(pkg-config --libs OGRE )
}
QMAKE_POST_LINK = cp \"$$README_FILE\" \"$$SOFA_DIR/lib/sofa-plugins\"
}

