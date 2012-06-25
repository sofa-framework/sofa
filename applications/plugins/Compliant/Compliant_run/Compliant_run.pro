load(sofa/pre)

TEMPLATE = app
TARGET = Compliant_run
CONFIG += console
DEFINES += SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD

SOURCES = Compliant_run.cpp

macx {
        CONFIG += app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
    QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
} else {
	RC_FILE = sofa.rc
}

unix {
    LIBS += -ldl
#    LIBS += -lFlexible -lCompliant
}

load(sofa/post)
