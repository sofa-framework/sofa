# Fichier genere par le module QMake de KDevelop. 
# -------------------------------------------------- 
# Sous dossier relatif au dossier principal du projet : ./applications/sofa/gui/viewer
# Cible : une bibliotheque:  sofaguiviewer$$LIBSUFFIX


HEADERS += viewer/SofaViewer.h \
           Main.h \
           RealGUI.h \
           GraphListenerQListView.h \
           GenGraphForm.h \
           AddObject.h \
           ModifyObject.h \
           WFloatLineEdit.h \
           FileManagement.h


SOURCES += Main.cpp \
           RealGUI_graph.cpp \
           RealGUI_record.cpp \
           RealGUI.cpp \
           GraphListenerQListView.cpp \
           GenGraphForm.cpp \
           AddObject.cpp \
           ModifyObject.cpp \
           WFloatLineEdit.cpp 


SOFA_DIR = ../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofaguiqt$$LIBSUFFIX

contains (DEFINES, SOFA_QT4) {	

	  CONFIG += $$CONFIGLIBRARIES qt uic uic3
	  QT += opengl qt3support xml
	  FORMS3 += GUI.ui
	  FORMS3 += BaseGenGraphForm.ui
	  FORMS3 += DialogAddObject.ui
}
else {
	  CONFIG += $$CONFIGLIBRARIES qt
	  QT += opengl	
	  FORMS += GUI.ui
	  FORMS += BaseGenGraphForm.ui
	  FORMS += DialogAddObject.ui
}

LIBS += $$SOFA_FRAMEWORK_LIBS $$SOFA_MODULES_LIBS
LIBS += -lsofagui$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS

contains( DEFINES, SOFA_GUI_QTVIEWER){

########################################################################
#  Qt
########################################################################
	SOURCES += viewer/qt/QtViewer.cpp
	HEADERS += viewer/qt/QtViewer.h
}


contains( DEFINES, SOFA_GUI_QGLVIEWER){
########################################################################
#  QGLViewer
########################################################################
	win32{
	  LIBS += $$SOFA_EXT_LIBS -lQGLViewer2
	}
	else{
	  LIBS += $$SOFA_EXT_LIBS -lQGLViewer
	}
	
	SOURCES += viewer/qgl/QtGLViewer.cpp
	HEADERS += viewer/qgl/QtGLViewer.h

}

contains( DEFINES, SOFA_GUI_QTOGREVIEWER){
########################################################################
#  OGRE 3D
########################################################################

	win32 {
		INCLUDEPATH += $(OGRE_HOME)/include
		QMAKE_LIBDIR += $(OGRE_HOME)/lib
		LIBS += OgreMain.lib
	}

	unix {
		macx:  QMAKE_CXXFLAGS += -Wno-unused
		
		!macx: {
                  	 QMAKE_CXXFLAGS += $$system(pkg-config --cflags OGRE )
		  LIBS += $$system(pkg-config --libs OGRE )
		  #CONFIG += link_pkgconfig
		  #PKGCONFIG += OGRE
		  #PKGCONFIG += CEGUI
		  #PKGCONFIG += OIS
                }
	}

        SOURCES += viewer/qtogre/DotSceneLoader.cpp \
                   viewer/qtogre/QtOgreViewer.cpp\ 
                   viewer/qtogre/QtOgreViewer_slots.cpp\ 
                   viewer/qtogre/OgreVisualModel.cpp \
                   viewer/qtogre/tinyxml.cpp \
                   viewer/qtogre/tinyxmlerror.cpp \
                   viewer/qtogre/tinyxmlparser.cpp
			   
	HEADERS += viewer/qtogre/DotSceneLoader.h \
                   viewer/qtogre/QtOgreViewer.h \
                   viewer/qtogre/OgreVisualModel.h \
                   viewer/qtogre/tinyxml.h
               
}
