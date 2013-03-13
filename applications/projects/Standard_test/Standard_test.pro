load(sofa/pre)

TEMPLATE = app
TARGET = Standard_test

DEFINES += SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD

CONFIG += console

LIBS += -lgtest \
		-lgtest_main
		
INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs

HEADERS += \
    Matrix_test.inl

SOURCES += \
    Matrix_test.cpp \
    OBBTest.cpp

win32 {
	QMAKE_CXXFLAGS_RELEASE += /MT
	QMAKE_CXXFLAGS_DEBUG += /MT
}

load(sofa/post)


