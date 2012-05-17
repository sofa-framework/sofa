load(sofa/pre)
defineAsPlugin(InvertibleFVM)

TARGET = InvertibleFVM

DEFINES += SOFA_BUILD_INVERTIBLEFVM

SOURCES = InvertibleFVMForceField.cpp \
          initPlugin.cpp

HEADERS = InvertibleFVMForceField.h \
	  InvertibleFVMForceField.inl \
	  initPlugin.h

README_FILE = InvertibleFVM.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
