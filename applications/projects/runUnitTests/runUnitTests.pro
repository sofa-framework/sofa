load(sofa/pre)


TEMPLATE = app
TARGET = runUnitTests

DEFINES += SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD

CONFIG += console

win32 {
	LIBS *= -lboost_unit_test_framework$$BOOST_SUFFIX
	LIBS *= -lboost_filesystem$$BOOST_SUFFIX
	QMAKE_CXXFLAGS -= -Zc:wchar_t-
	QMAKE_CXXFLAGS += -Zc:wchar_t
}

macx {
	CONFIG += app_bundle
	RC_FILE = runSOFA.icns
	QMAKE_INFO_PLIST = Info.plist
    QMAKE_BUNDLE_DATA += $$APP_BUNDLE_DATA
}

unix {
   LIBS += -ldl
   LIBS *= -l$${BOOST_PREFIX}boost_unit_test_framework$$BOOST_SUFFIX
   LIBS *= -l$${BOOST_PREFIX}boost_filesystem$$BOOST_SUFFIX
}

SOURCES = \
    runner.cpp



#unix: QMAKE_LIBDIR += $$SOFA_GPU_CUDA_DIR



load(sofa/post)
