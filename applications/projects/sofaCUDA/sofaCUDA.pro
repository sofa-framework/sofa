SOFA_DIR=../../..
TEMPLATE = app

include($$SOFA_DIR/sofa.cfg)

TARGET = sofaCUDA$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECT \
          warn_on

SOURCES = Main.cpp 
HEADERS = 

RC_FILE = sofa.rc

contains (DEFINES, SOFA_GUI_QT) {
CONFIG += qt
QT += opengl qt3support
}

LIBS += -lsofahelper$$LIBSUFFIX -lsofadefaulttype$$LIBSUFFIX -lsofacore$$LIBSUFFIX -lNewMat$$LIBSUFFIX -lSLC$$LIBSUFFIX
LIBS += -lsofacomponent$$LIBSUFFIX -lsofasimulation$$LIBSUFFIX
LIBS += -lsofagpucuda$$LIBSUFFIX

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

########################################################################
#  CUDA
########################################################################
win32 {
  INCLUDEPATH += $(CUDA_INC_DIR)
  QMAKE_LIBDIR += $(CUDA_LIB_DIR)
  LIBS += -lcudart

  cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
  cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
}
unix {
  # auto-detect CUDA path
  CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
  INCLUDEPATH += $$CUDA_DIR/include
  QMAKE_LIBDIR += $$CUDA_DIR/lib
  LIBS += -lcudart

  cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
  cuda.commands = nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
  cuda.depends = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'
}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_UNIX_COMPILERS += cuda

########################################################################
