SOFA_DIR=../../..
TEMPLATE = app

include($${SOFA_DIR}/sofa.cfg)

TARGET = mixedPendulum$$SUFFIX
DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTGUI
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
    QMAKE_POST_LINK = ln -sf mixedPendulum$$SUFFIX $$DESTDIR/mixedPendulum-latest
}


SOURCES = Main.cpp 
HEADERS = 
