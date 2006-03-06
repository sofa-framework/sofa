include(../../../sofa.cfg)
TEMPLATE = $$TEMPLATEAPPPROJECT
CONFIG =  $$CONFIGPROJECT \
          warn_on \
          $$CONFIGDEBUG

DESTDIR = . # ../../bin
TARGET = example1
OBJECTS_DIR = ./OBJ
INCLUDEPATH = ../..
INCLUDEPATH += ../../../include
DEPENDPATH = ../..
SOURCES = Main.cpp 
HEADERS = 

win32{
  LIBS = -lSofaAbstract -lSofaCore -lSofaComponents -lSofaGUIFLTK -llibxml2 -lfltk -lfltkgl -lGLaux -lglut32 -lcomctl32 
  contains (CONFIGPROJECT, vc7) {
	contains (CONFIGPROJECT, debug) {
	  	LIBS += -NODEFAULTLIB:libcd 	
	}	
	contains (CONFIGPROJECT, release) {
	  	LIBS += -NODEFAULTLIB:libc
	}
  }
  QMAKE_LIBDIR = ../../../lib/$$LIBSDIRECTORY
  #QMAKE_LFLAGS = 
  #QMAKE_LIBS_WINDOWS = ""
  #QMAKE_CXXFLAGS += -GR -GX
  DEFINES = WIN32
}

unix {
  QMAKE_CXXFLAGS += -g
  LIBS = -L../../../lib/$$LIBSDIRECTORY -L/usr/X11R6/lib -lSofaAbstract -lSofaCore -lSofaComponents -lSofaGUIFLTK -lglut -lGL -lGLU -lfltk_gl -lz -lfltk -lpthread -lxml2 -lz

#-lManager -lVisualModel -lGUI  -lglut -lGL -lGLU  -lfltk_gl -lCollisionResponse -lBehaviorModel -lCollisionModel -lCollisionDetection -lz -lCommon -lfltk -lThread -lpthread -lIntegrationSolver -lxml2 -lz
}
