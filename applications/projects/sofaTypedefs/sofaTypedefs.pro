load(sofa/pre)

TEMPLATE = app
TARGET = sofaTypedefs

CONFIG += console 

SOURCES = Main.cpp 

contains (CONFIGDEBUG, debug) {
	QMAKE_POST_LINK = call ..\..\..\bin\sofaTypedefsd.exe
}
	
contains (CONFIGDEBUG, release) {
	QMAKE_POST_LINK = call ..\..\..\bin\sofaTypedefs.exe
}


load(sofa/post)
