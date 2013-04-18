load(sofa/pre)

TEMPLATE = app
TARGET = Debug_test

CONFIG += console

LIBS += -lgtest -lgtest_main
		
INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/extlibs
INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/extlibs/gtest/include

QMAKE_LIBDIR += $$SOFA_INSTALL_INC_DIR/extlibs/gtest/lib

HEADERS +=

SOURCES += \
 My_test.cpp \


win32 {
	QMAKE_CXXFLAGS_RELEASE += /MT
	QMAKE_CXXFLAGS_DEBUG += /MT
}


load(sofa/post)


