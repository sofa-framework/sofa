load(sofa/pre)

TEMPLATE = app
TARGET = sofaTypedefs

CONFIG += console 

SOURCES = Main.cpp 

contains (CONFIGDEBUG, debug) {
    win32 {
      QMAKE_POST_LINK = call ..\\..\\..\\bin\\sofaTypedefsd.exe
    }
    else:macx {
      QMAKE_POST_LINK = export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../../lib:../../../lib/macx && ../../../bin/sofaTypedefsd
    }
    else:unix {
      QMAKE_POST_LINK = ../../../bin/sofaTypedefsd
    }
}
	
contains (CONFIGDEBUG, release) {
    win32 {
      QMAKE_POST_LINK = call ..\\..\\..\\bin\\sofaTypedefs.exe
    }
    else:macx {
      QMAKE_POST_LINK = export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:../../../lib:../../../lib/macx && ../../../bin/sofaTypedefs
    }
    else:unix {
      QMAKE_POST_LINK = ../../../bin/sofaTypedefs
    }
}

load(sofa/post)
