load(sofa/pre)

TEMPLATE = app
TARGET = Compliant_test

CONFIG += console

LIBS += -lsofagraph$$LIBSUFFIX \
		-lgtest \
		-lgtest_main

INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/extlibs
INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/extlibs/gtest/include

QMAKE_LIBDIR += $$SOFA_INSTALL_INC_DIR/extlibs/gtest/lib
		
SOURCES = Compliant_test.cpp

macx {
	CONFIG += app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
    QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
}

unix {
    LIBS += -ldl
    INCLUDEPATH += /usr/include/suitesparse/
}

win32 {
	INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs/SuiteSparse/cholmod/Include
	QMAKE_LIBDIR += $$SOFA_INSTALL_INC_DIR/extlibs/SuiteSparse/cholmod/Lib
	
	QMAKE_CXXFLAGS_RELEASE += /MT
	QMAKE_CXXFLAGS_DEBUG += /MT
}

load(sofa/post)
