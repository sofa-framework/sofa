include(../../../sofa.cfg)
TEMPLATE = app
CONFIG += $$CONFIGPROJECT \
          warn_on

DESTDIR = ../../../bin
TARGET = SofaFlowVR$$SUFFIX

OBJECTS_DIR = OBJ/$$CONFIGDEBUG
INCLUDEPATH = ../..
INCLUDEPATH += ../../../include
DEPENDPATH = ../..

SOURCES = Main.cpp 
HEADERS = 

contains (DEFINES, SOFA_GUI_QT) {
CONFIG += qt
QT += opengl qt3support
}

QMAKE_LIBDIR = ../../../lib/$$LIBSDIRECTORY ../../../lib/$$LIBSDIRECTORY/../Common
LIBS = -lSofaAbstract$$LIBSUFFIX -lSofaCore$$LIBSUFFIX -lSofaComponents$$LIBSUFFIX -lNewMat$$LIBSUFFIX

########################################################################
#  FLOWVR
########################################################################
FLOWVR = $(FLOWVR_PREFIX)

QMAKE_CXXFLAGS += `pkg-config --cflags flowvr-mod flowvr-ftl flowvr-render`
QMAKE_LDFLAGS += `pkg-config --libs flowvr-mod flowvr-ftl flowvr-render`
QMAKE_LFLAGS_DEBUG+= `pkg-config --libs flowvr-mod flowvr-ftl flowvr-render`
QMAKE_LFLAGS_RELEASE+= `pkg-config --libs flowvr-mod flowvr-ftl flowvr-render`
#contains (CONFIGDEBUG, debug) {
#    DEFINES += FLOWVR_DEBUG DEBUG VERBOSE_ENABLE
#}

########################################################################

win32{
  LIBS += -llibxml2 -lGLaux -lglut32 -lcomctl32 -lopengl32 -lglu32 -lAdvAPI32 -lUser32 -lShell32 -lGdi32 -lWSock32 -lWS2_32 -lOle32
  contains (DEFINES, SOFA_GUI_FLTK) {
	LIBS += -lSofaGUIFLTK$$LIBSUFFIX -lfltk -lfltkgl
  }
  contains (DEFINES, SOFA_GUI_QT) {
	LIBS += -lSofaGUIQT$$LIBSUFFIX
  }
  contains (CONFIGPROJECT, vc7) {
	contains (CONFIGDEBUG, debug) {
	  	QMAKE_LFLAGS += /NODEFAULTLIB:libcd /NODEFAULTLIB:MSVCRT	
	}	
	contains (CONFIGDEBUG, release) {
	  	QMAKE_LFLAGS += /NODEFAULTLIB:libc /NODEFAULTLIB:MSVCRTD
	}
  }
  #QMAKE_LFLAGS = 
  #QMAKE_LIBS_WINDOWS = ""
  #QMAKE_CXXFLAGS += -GR -GX
  #DEFINES = WIN32
}

unix {
  LIBS += -L/usr/X11R6/lib -lglut -lGL -lGLU -lpthread -lxml2 -lz
  contains (DEFINES, SOFA_GUI_FLTK) {
	LIBS += -lSofaGUIFLTK$$LIBSUFFIX -lfltk_gl -lfltk
  }
  contains (DEFINES, SOFA_GUI_QT) {
	LIBS += -lSofaGUIQT$$LIBSUFFIX
  }
}
