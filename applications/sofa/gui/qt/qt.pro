# Fichier genere par le module QMake de KDevelop. 
# -------------------------------------------------- 
# Sous dossier relatif au dossier principal du projet : ./applications/sofa/gui/viewer
# Cible : une bibliotheque:  sofaguiviewer$$LIBSUFFIX

SOFA_DIR = ../../../..
TEMPLATE = lib
TARGET = sofaguiqt

include($${SOFA_DIR}/sofa.cfg)


DEFINES += SOFA_BUILD_SOFAGUIQT


LIBS += $$SOFA_FRAMEWORK_LIBS
LIBS += $$SOFA_MODULES_LIBS
LIBS += -lsofagui$$LIBSUFFIX
LIBS += $$SOFA_GUI_EXT_LIBS
LIBS += $$SOFA_EXT_LIBS


contains (DEFINES, SOFA_QT4) {	

	  CONFIG += $$CONFIGLIBRARIES qt uic uic3
	  !contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
      CONFIG += dll
}
	  QT += opengl qt3support xml
	  FORMS3 += GUI.ui
	  FORMS3 += BaseGenGraphForm.ui
	  FORMS3 += DialogAddObject.ui
	  FORMS3 += PluginManager.ui
	  FORMS3 += MouseManager.ui
  	  FORMS3 += VideoRecorderManager.ui
contains (DEFINES, SOFA_DUMP_VISITOR_INFO){
	  FORMS3 += VisitorGUI.ui
	}
}
else {
	  CONFIG += $$CONFIGLIBRARIES qt
	  !contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
      CONFIG += dll
}
	  QT += opengl	
	  FORMS += GUI.ui
	  FORMS += BaseGenGraphForm.ui
	  FORMS += DialogAddObject.ui
	  FORMS += PluginManager.ui
	  FORMS += MouseManager.ui
  	  FORMS += VideoRecorderManager.ui
contains (DEFINES, SOFA_DUMP_VISITOR_INFO){
	  FORMS += VisitorGUI.ui
}
}


HEADERS += viewer/VisualModelPolicy.h \
	   viewer/SofaViewer.h \
           viewer/ViewerFactory.h \
           GraphListenerQListView.h \
           GenGraphForm.h \
           AddObject.h \
           RealGUI.h \
	   DataWidget.h \
	   DataFilenameWidget.h \
           DisplayFlagWidget.h \
           GraphDataWidget.h \
	   MaterialDataWidget.h \
           ModifyObject.h \
           SimpleDataWidget.h \
	   SofaGUIQt.h \
           StructDataWidget.h \
           TableDataWidget.h \
           WDoubleLineEdit.h \ 
           FileManagement.h \
           SofaPluginManager.h \
           SofaMouseManager.h \
           SofaVideoRecorderManager.h \	
		   PickHandlerCallBacks.h \
           QDataDescriptionWidget.h \
	   QDisplayDataWidget.h \     
           QEnergyStatWidget.h \              
           QTabulationModifyObject.h \
           QTransformationWidget.h \
           QMouseOperations.h \
	   QSofaListView.h \
	   QSofaRecorder.h \
	   QSofaStatWidget.h \
           QMenuFilesRecentlyOpened.h \
           ImageQt.h \ 
    



SOURCES += viewer/SofaViewer.cpp \
           viewer/ViewerFactory.cpp \
           GraphListenerQListView.cpp \
           GenGraphForm.cpp \
           AddObject.cpp \
           RealGUI.cpp \
	   DataWidget.cpp \ 
	   DataFilenameWidget.cpp \
           DisplayFlagWidget.cpp \
           GraphDataWidget.cpp \  
	   MaterialDataWidget.cpp \
           ModifyObject.cpp \
           SimpleDataWidget.cpp \
           StructDataWidget.cpp \
           TableDataWidget.cpp \
           WDoubleLineEdit.cpp \
           FileManagement.cpp \
           SofaPluginManager.cpp \
           SofaVideoRecorderManager.cpp \
           SofaMouseManager.cpp \
           QDataDescriptionWidget.cpp \
		   PickHandlerCallBacks.cpp \
	   QDisplayDataWidget.cpp \
           QEnergyStatWidget.cpp \       
	   QMouseOperations.cpp \               
           QTabulationModifyObject.cpp \
           QTransformationWidget.cpp \
	   QSofaListView.cpp \
	   QSofaRecorder.cpp \
	   QSofaStatWidget.cpp \
           QMenuFilesRecentlyOpened.cpp \
           ImageQt.cpp \ 


contains (DEFINES, SOFA_QT4){
HEADERS += QModelViewTableDataContainer.h \
           QModelViewTableUpdater.h
}

!contains (DEFINES, SOFA_QT4){
HEADERS += QTableDataContainer.h \
           QTableUpdater.h
}

contains (DEFINES, SOFA_DUMP_VISITOR_INFO){
HEADERS += GraphVisitor.h \
           WindowVisitor.h \
           QVisitorControlPanel.h \
           PieWidget.h

SOURCES += GraphVisitor.cpp \
           WindowVisitor.cpp \
           QVisitorControlPanel.cpp \
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
	  LIBS += $$SOFA_EXT_LIBS -lQGLViewer$$LIBSUFFIX
	}
	else{
	  LIBS += $$SOFA_EXT_LIBS -lQGLViewer$$LIBSUFFIX
	}
	
	SOURCES += viewer/qgl/QtGLViewer.cpp
	HEADERS += viewer/qgl/QtGLViewer.h

}
