load(sofa/pre)

TEMPLATE = lib
TARGET = sofamodeler

CONFIGSTATIC = static

INCLUDEPATH *= $ROOT_SRC_DIR/applications/sofa/gui/qt#/$$UI_DIR # HACK: this library uses some uic generated headers from this directory.
DEPENDPATH *= $ROOT_SRC_DIR/applications/sofa/gui/qt/$$UI_DIR # HACK: this library uses some uic generated headers from this directory.

SOURCES = SofaModeler.cpp \
          GraphModeler.cpp \
		  LinkComponent.cpp \
          SofaTutorialManager.cpp \
          TutorialSelector.cpp \
          AddPreset.cpp \
          FilterLibrary.cpp \
          GlobalModification.cpp \
          GraphHistoryManager.cpp \
          ModifierCondition.cpp \
          ../../../sofa/gui/qt/SofaPluginManager.cpp 

HEADERS = SofaModeler.h \
          GraphModeler.h \
          SofaTutorialManager.h \
          TutorialSelector.h \
          AddPreset.h \
          FilterLibrary.h \
          GlobalModification.h \
		  LinkComponent.h \
          GraphHistoryManager.h \
          ModifierCondition.h \
          ../../../sofa/gui/qt/SofaPluginManager.h 

contains (DEFINES, SOFA_QT4) {	

	HEADERS += QSofaTreeLibrary.h \
	           QCategoryTreeLibrary.h \ 
	           QComponentTreeLibrary.h
	SOURCES += QSofaTreeLibrary.cpp \
	           QCategoryTreeLibrary.cpp \ 
	           QComponentTreeLibrary.cpp
                     
	CONFIG += qt uic uic3
	QT += qt3support xml
	FORMS3 += Modeler.ui 
	FORMS3 += DialogAddPreset.ui
	FORMS3 += ../../../sofa/gui/qt/PluginManager.ui
}
else {

  HEADERS += QSofaLibrary.h \
             QCategoryLibrary.h \ 
             QComponentLibrary.h
  SOURCES += QSofaLibrary.cpp \
             QCategoryLibrary.cpp \ 
             QComponentLibrary.cpp
                     
	CONFIG += qt
	FORMS += Modeler.ui
	FORMS += DialogAddPreset.ui
	FORMS += ../../../sofa/gui/qt/PluginManager.ui                    
}

load(sofa/post)
