load(sofa/pre)
defineAsPlugin(Python)

TARGET = Python

DEFINES += SOFA_BUILD_PYTHON

SOURCES = initPython.cpp

HEADERS = initPython.h

README_FILE = Python.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
