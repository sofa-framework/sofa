load(sofa/pre)

TEMPLATE = app
TARGET = Flexible_test

CONFIG += console

LIBS += -lgtest \
		-lgtest_main


INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/extlibs
INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/extlibs/gtest/include
INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/applications/projects/Standard_test

QMAKE_LIBDIR += $$SOFA_INSTALL_INC_DIR/extlibs/gtest/lib


SOURCES += StrainMappings_test.cpp


unix {
    LIBS += -ldl
}

win32 {
	QMAKE_CXXFLAGS_RELEASE += /MT
	QMAKE_CXXFLAGS_DEBUG += /MT
}

load(sofa/post)
