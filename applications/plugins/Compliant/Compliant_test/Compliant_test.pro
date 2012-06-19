load(sofa/pre)


TEMPLATE = app
TARGET = Compliant_test

DEFINES += SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD

CONFIG += console

macx {
        CONFIG += app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
    QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
} else {
	RC_FILE = sofa.rc
}

unix {
#   LIBS += -ldl
LIBS *= -lboost_unit_test_framework

}

SOURCES = Compliant_test.cpp


load(sofa/post)
