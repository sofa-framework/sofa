load(sofa/pre)
defineAsPlugin(ColladaSceneLoader)

TARGET = ColladaSceneLoader

DEFINES += SOFA_BUILD_COLLADASCENELOADER

SOURCES = SceneColladaLoader.cpp \
		  initPlugin.cpp

HEADERS = SceneColladaLoader.h \
		  initPlugin.h

README_FILE = ColladaSceneLoader.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

win32 {
	LIBS += assimp/lib/assimp.lib
}
unix {
    INCLUDEPATH *= assimp/
    LIBS += -lassimp
}

load(sofa/post)
