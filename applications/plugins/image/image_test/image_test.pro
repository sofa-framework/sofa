load(sofa/pre)

TEMPLATE = app
TARGET = image_test

CONFIG += console

LIBS += -lgtest \
		-lgtest_main

INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs
		
HEADERS += BranchingImage_test.inl

SOURCES += BranchingImage_test.cpp

contains(DEFINES, SOFA_IMAGE_HAVE_OPENCV) { # should be "SOFA_HAVE_OPENCV" -> use "SOFA_IMAGE_HAVE_OPENCV" until the opencv plugin is fixed..
    INCLUDEPATH += $$SOFA_OPENCV_PATH
        LIBS += -lml  -lcvaux -lhighgui -lcv -lcxcore
        }

contains(DEFINES, SOFA_HAVE_LIBFREENECT) {
        INCLUDEPATH += $$SOFA_LIBFREENECT_PATH
        LIBS += -lfreenect -lfreenect_sync
        }


unix {
    LIBS += -ldl
}

win32 {
	QMAKE_CXXFLAGS_RELEASE += /MT
	QMAKE_CXXFLAGS_DEBUG += /MT
}

load(sofa/post)


