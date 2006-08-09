include(../../../sofa.cfg)
TEMPLATE = app
CONFIG +=  $$CONFIGPROJECT \
	      console

DESTDIR = ../../../bin
TARGET = GenerateRigid$$SUFFIX
OBJECTS_DIR = ./OBJ/$$CONFIGDEBUG
INCLUDEPATH = ../..
INCLUDEPATH += ../../../include
DEPENDPATH = ../..
SOURCES = GenerateRigid.cpp \
	      Main.cpp
HEADERS = GenerateRigid.h

QMAKE_LIBDIR = ../../../lib/$$LIBSDIRECTORY ../../../lib/$$LIBSDIRECTORY/../Common
LIBS = -lSofaAbstract$$LIBSUFFIX -lSofaCore$$LIBSUFFIX -lSofaComponents$$LIBSUFFIX

win32{
  LIBS += -llibxml2 -lGLaux -lglut32 -lcomctl32 -lopengl32 -lglu32 -lAdvAPI32 -lUser32 -lShell32 -lGdi32 -lWSock32 -lWS2_32 -lOle32
  contains (CONFIGPROJECT, vc7) {
	contains (CONFIGDEBUG, debug) {
	  	QMAKE_LFLAGS += /NODEFAULTLIB:libcd /NODEFAULTLIB:MSVCRT	
	}	
	contains (CONFIGDEBUG, release) {
	  	QMAKE_LFLAGS += /NODEFAULTLIB:libc /NODEFAULTLIB:MSVCRTD
	}
  }
}

unix {
  LIBS += -L/usr/X11R6/lib -lglut -lGL -lGLU -lpthread -lxml2 -lz
}
