load(sofa/pre)
defineAsPlugin(SixenseHydra)

TARGET = SixenseHydra

DEFINES += SOFA_BUILD_SIXENSE_HYDRA

SOURCES = initSixenseHydra.cpp \
		  RazerHydraDriver.cpp

HEADERS = initSixenseHydra.h \
		  RazerHydraDriver.h

README_FILE = SixenseHydra.txt

#TODO: add an install target for README files

win32 {
	INCLUDEPATH += $$ROOT_SRC_DIR/applications/plugins/SixenseHydra/include
	LIBS += -L$$ROOT_SRC_DIR/applications/plugins/SixenseHydra/lib/win32/release_dll -lsixense
	LIBS += -L$$ROOT_SRC_DIR/applications/plugins/SixenseHydra/lib/win32/release_dll -lsixense_utils
}

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)