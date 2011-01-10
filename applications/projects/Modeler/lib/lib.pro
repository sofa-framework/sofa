SOFA_DIR=../../../..
TEMPLATE = lib
TARGET = sofamodeler

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGLIBRARIES
# LIBS += -lsofasimulation$$LIBSUFFIX 
contains(CONFIGSTATIC, static) {
#LIBS -= $${SOFA_EXT_LIBS}
LIBS += $${SOFA_MODULES_LIBS}
LIBS += $${SOFA_FRAMEWORK_LIBS}
}
else {
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS
LIBS += $$SOFA_EXT_LIBS

INCLUDEPATH += $$SOFA_DIR/extlibs
}

SOURCES = SofaModeler.cpp \
          GraphModeler.cpp \
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
                     
	  CONFIG += $$CONFIGLIBRARIES qt uic uic3
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
                     
	  CONFIG += $$CONFIGLIBRARIES qt
	  FORMS += Modeler.ui
	  FORMS += DialogAddPreset.ui
        FORMS += ../../../sofa/gui/qt/PluginManager.ui                    
}


#add local libraries to the modeler
exists(lib-local.cfg): include(lib-local.cfg)
