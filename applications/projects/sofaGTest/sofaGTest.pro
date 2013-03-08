load(sofa/pre)

TEMPLATE = app
TARGET = sofaGTest

DEFINES += SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD

CONFIG += console

HEADERS += \
    initSofaGTest.h \
    Matrix_test.inl

SOURCES += \
    initSofaGTest.cpp \
    Matrix_test.cpp \
    OBBTest.cpp
 
LIBS += -lgtest
LIBS += -lgtest_main

DEFINES += SOFA_BUILD_SOFA_GTEST

win32 {
	QMAKE_CXXFLAGS_RELEASE += /MT
	QMAKE_CXXFLAGS_DEBUG += /MT
}

load(sofa/post)


