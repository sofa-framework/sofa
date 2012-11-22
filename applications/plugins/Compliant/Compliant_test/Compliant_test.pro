load(sofa/pre)

TEMPLATE = lib

CONFIG += console
DEFINES += SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD

SOURCES = Compliant_test.cpp

macx {
        CONFIG += app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
    QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
}

unix {
    LIBS += -ldl
    LIBS *= -lboost_unit_test_framework
    INCLUDEPATH += /usr/include/suitesparse/
}

win32 {
	INCLUDEPATH += $$SOFA_INSTALL_INC_DIR\extlibs\SuiteSparse\cholmod\Include
	QMAKE_LIBDIR += $$SOFA_INSTALL_INC_DIR\extlibs\SuiteSparse\cholmod\Lib
}

load(sofa/post)
