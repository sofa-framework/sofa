load(sofa/pre)

TEMPLATE = app
TARGET = Compliant_run
CONFIG += console

LIBS += -lsofagraph

SOURCES = Compliant_run.cpp

macx {
        CONFIG += app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
    QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
}

unix {
	RC_FILE = sofa.rc
    LIBS += -ldl
#    LIBS += -lFlexible -lCompliant
    INCLUDEPATH += /usr/include/suitesparse/

}

win32 {
	INCLUDEPATH += $$SOFA_INSTALL_INC_DIR\extlibs\SuiteSparse\cholmod\Include
	QMAKE_LIBDIR += $$SOFA_INSTALL_INC_DIR\extlibs\SuiteSparse\cholmod\Lib
}

load(sofa/post)
