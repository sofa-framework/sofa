load(sofa/pre)


TEMPLATE = app
TARGET = Compliant_test
DEFINES *= EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

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
}

SOURCES = Compliant_test.cpp
HEADERS = 

#unix: QMAKE_LIBDIR += $$SOFA_GPU_CUDA_DIR

load(sofa/post)
