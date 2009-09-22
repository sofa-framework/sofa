# Fichier genere par le module QMake de KDevelop. 
# -------------------------------------------------- 
# Sous dossier relatif au dossier principal du projet : ./applications/sofa/gui/viewer
# Cible : une bibliotheque:  sofaguiviewer$$LIBSUFFIX


SOFA_DIR = ../../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofaguiqt$$LIBSUFFIX

LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += $$SOFA_MODULES_LIBS
LIBS += -lsofagui$$LIBSUFFIX
LIBS += $$SOFA_GUI_EXT_LIBS
LIBS += $$SOFA_EXT_LIBS


contains (DEFINES, SOFA_QT4) {	

	  CONFIG += $$CONFIGLIBRARIES qt uic uic3
	  QT += opengl qt3support xml
	  FORMS3 += GUI.ui
	  FORMS3 += BaseGenGraphForm.ui
	  FORMS3 += DialogAddObject.ui
	  FORMS3 += PluginManager.ui
	  FORMS3 += MouseManager.ui
contains (DEFINES, SOFA_DUMP_VISITOR_INFO){
	  FORMS3 += VisitorGUI.ui
}
}
else {
	  CONFIG += $$CONFIGLIBRARIES qt
	  QT += opengl	
	  FORMS += GUI.ui
	  FORMS += BaseGenGraphForm.ui
	  FORMS += DialogAddObject.ui
	  FORMS += PluginManager.ui
	  FORMS += MouseManager.ui
contains (DEFINES, SOFA_DUMP_VISITOR_INFO){
	  FORMS += VisitorGUI.ui
}
}


HEADERS += viewer/SofaViewer.h \
           GraphListenerQListView.h \
           GenGraphForm.h \
           AddObject.h \
           Main.h \
           RealGUI.h \
           DisplayFlagWidget.h \
           GraphDataWidget.h \
           ModifyObject.h \
           SimpleDataWidget.h \
           StructDataWidget.h \
           TableDataWidget.h \
           WFloatLineEdit.h \ 
           FileManagement.h \
           SofaPluginManager.h \
           SofaMouseManager.h \
           QMouseOperations.h \
           ImageQt.h

SOURCES += Main.cpp \
           GraphListenerQListView.cpp \
           GenGraphForm.cpp \
           AddObject.cpp \
           RealGUI_graph.cpp \
           RealGUI_record.cpp \
           RealGUI.cpp \
           DisplayFlagWidget.cpp \
           GraphDataWidget.cpp \  
           ModifyObject.cpp \
           SimpleDataWidget.cpp \
           StructDataWidget.cpp \
           TableDataWidget.cpp \
           WFloatLineEdit.cpp \
           FileManagement.cpp \
           SofaPluginManager.cpp \
           SofaMouseManager.cpp \
           QMouseOperations.cpp \
           ImageQt.cpp

contains (DEFINES, SOFA_DUMP_VISITOR_INFO){
HEADERS += GraphVisitor.h \
           WindowVisitor.h \
           PieWidget.h

SOURCES += GraphVisitor.cpp \
           WindowVisitor.cpp \
           PieWidget.cpp
}

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
# why do we include lQGLViewer2 whereas in extlibs, QGLviewer creates  llQGLViewer$$LIBSUFFIX ???
#	  LIBS += $$SOFA_EXT_LIBS -lQGLViewer2$$LIBSUFFIX
	  LIBS += $$SOFA_EXT_LIBS -lQGLViewer$$LIBSUFFIX
	}
	else{
	  LIBS += $$SOFA_EXT_LIBS -lQGLViewer$$LIBSUFFIX
	}
	
	SOURCES += viewer/qgl/QtGLViewer.cpp
	HEADERS += viewer/qgl/QtGLViewer.h

}

########################################################################
#  OGRE 3D
########################################################################

contains( DEFINES, SOFA_GUI_QTOGREVIEWER){

        SOURCES += viewer/qtogre/DotSceneLoader.cpp \
                   viewer/qtogre/QtOgreViewer.cpp\ 
                   viewer/qtogre/QtOgreViewer_slots.cpp\ 
                   viewer/qtogre/OgreVisualModel.cpp \
                   viewer/qtogre/QOgreLightWidget.cpp
			   
	HEADERS += viewer/qtogre/DotSceneLoader.h \
                   viewer/qtogre/QtOgreViewer.h \
                   viewer/qtogre/OgreVisualModel.h \
                   viewer/qtogre/QOgreLightWidget.h
               
}
