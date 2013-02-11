load(sofa/pre)

TEMPLATE = lib

CONFIG += console

DEFINES += SOFA_BUILD_Image_test

HEADERS += initImage_test.h \
           BranchingImage_test.inl \


SOURCES += initImage_test.cpp \
           BranchingImage_test.cpp \





INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs

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
    LIBS *= -l$${BOOST_PREFIX}boost_unit_test_framework$$BOOST_SUFFIX
}

win32 {
	LIBS *= -lboost_unit_test_framework$$BOOST_SUFFIX
}


load(sofa/post)


