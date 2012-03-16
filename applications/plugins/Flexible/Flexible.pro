load(sofa/pre)
defineAsPlugin(Flexible)

TARGET = Flexible

DEFINES += SOFA_BUILD_Flexible

SOURCES = initFlexible.cpp

HEADERS = initFlexible.h

README_FILE = Flexible.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
