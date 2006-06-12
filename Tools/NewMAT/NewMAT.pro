include(../../sofa.cfg)
TEMPLATE = lib
CONFIG += $$CONFIGLIBRARIES
DESTDIR = ../../lib/$$LIBSDIRECTORY
TARGET = NewMat
OBJECTS_DIR = OBJ/$$CONFIGDEBUG
INCLUDEPATH = ../..
INCLUDEPATH += ../../include
DEPENDPATH = ../..

use_namespace 

#LIBS = -lSofaAbstract
#QMAKE_LIBDIR = ../../../lib/$$LIBSDIRECTORY ../../../lib/$$LIBSDIRECTORY/../Common

SOURCES = \    
        newmat1.cpp \
        newmat2.cpp \
        newmat3.cpp \
        newmat4.cpp \
        newmat5.cpp \
        newmat6.cpp \
        newmat7.cpp \
        newmat8.cpp \
        newmat9.cpp \
        newmatex.cpp \
        newmatrm.cpp \
        bandmat.cpp \
        submat.cpp \
        myexcept.cpp \
        cholesky.cpp \
        evalue.cpp \
        fft.cpp \
        hholder.cpp \
        jacobi.cpp \
        newfft.cpp \
        sort.cpp \
        svd.cpp 

HEADERS = \
        include.h \
        newmat.h \
        newmatrc.h \
        boolean.h \
        myexcept.h \
        controlw.h \
		newmatap.h \
		newmatrm.h \
		precisio.h 
