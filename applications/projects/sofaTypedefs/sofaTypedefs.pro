load(sofa/pre)

TEMPLATE = app
TARGET = sofaTypedefs

CONFIG += console 

SOURCES = Main.cpp 

contains (CONFIGDEBUG, debug) {
	unix: QMAKE_POST_LINK = ../../../bin/sofaTypedefsd
	win32: QMAKE_POST_LINK = call ..\\..\\..\\bin\\sofaTypedefsd.exe
}
	
contains (CONFIGDEBUG, release) {
	unix: QMAKE_POST_LINK = ../../../bin/sofaTypedefs
	win32: QMAKE_POST_LINK = call ..\\..\\..\\bin\\sofaTypedefs.exe
}


load(sofa/post)
