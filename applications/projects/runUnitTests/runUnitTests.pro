load(sofa/pre)


TEMPLATE = app
TARGET = UnitTests

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
}

SOURCES = runUnitTests.cpp \
    MatrixTest.cpp
HEADERS = \
    MatrixTest.h \
    MatrixTest.inl \
    EigenBaseSparseMatrixNew.h \
    EigenSparseMatrixNew.h \

#unix: QMAKE_LIBDIR += $$SOFA_GPU_CUDA_DIR

load(sofa/post)
