load(sofa/pre)


TEMPLATE = app
TARGET = runUnitTests

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
   LIBS += -ldl
   LIBS *= -l$${BOOST_PREFIX}boost_unit_test_framework$$BOOST_SUFFIX
}

SOURCES = \
    runner.cpp



#unix: QMAKE_LIBDIR += $$SOFA_GPU_CUDA_DIR



load(sofa/post)
