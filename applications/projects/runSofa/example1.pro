SOFA_DIR=../../..
TEMPLATE = app

include($$SOFA_DIR/sofa.cfg)

CONFIG += $$CONFIGPROJECT \
          warn_on

DESTDIR = $$SOFA_DIR/bin
TARGET = runSofa$$SUFFIX
RC_FILE = sofa.rc

SOURCES = Main.cpp 
HEADERS = 

contains (DEFINES, SOFA_GUI_QT) {
CONFIG += qt
QT += opengl qt3support
}

LIBS = -lsofahelper$$LIBSUFFIX -lsofadefaulttype$$LIBSUFFIX -lsofacore$$LIBSUFFIX -lNewMat$$LIBSUFFIX

win32{
  LIBS += -llibxml2 -lGLaux -lglut32 -lcomctl32 -lopengl32 -lglu32 -lAdvAPI32 -lUser32 -lShell32 -lGdi32 -lWSock32 -lWS2_32 -lOle32
  contains (DEFINES, SOFA_GUI_FLTK) {
	LIBS += -lsofaguifltk$$LIBSUFFIX -lfltk -lfltkgl
  }
  contains (DEFINES, SOFA_GUI_QT) {
	LIBS += -lsofaguiqt$$LIBSUFFIX
  }
  contains (CONFIGPROJECT, vc7) {
	contains (CONFIGDEBUG, debug) {
	  	QMAKE_LFLAGS += /NODEFAULTLIB:libcd /NODEFAULTLIB:MSVCRT	
	}	
	contains (CONFIGDEBUG, release) {
	  	QMAKE_LFLAGS += /NODEFAULTLIB:libc /NODEFAULTLIB:MSVCRTD
	}
  }
#  LIBS += -lSofaContribFluidGrid
#  QMAKE_LFLAGS = /INCLUDE:_class_Fluid2D /INCLUDE:_class_Fluid3D
  #QMAKE_LIBS_WINDOWS = ""
  #QMAKE_CXXFLAGS += -GR -GX
  #DEFINES = WIN32
}

unix {
  LIBS += -L/usr/X11R6/lib -lglut -lGL -lGLU -lpthread -lxml2 -lz
  contains (DEFINES, SOFA_GUI_FLTK) {
	LIBS += -lsofaguifltk$$LIBSUFFIX -lfltk_gl -lfltk
  }
  contains (DEFINES, SOFA_GUI_QT) {
	LIBS += -lsofaguiqt$$LIBSUFFIX
  }
}
