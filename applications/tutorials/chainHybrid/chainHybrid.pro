SOFA_DIR=../../..
TEMPLATE = app
TARGET = chainHybrid

include($${SOFA_DIR}/sofa.cfg)

#Uncomment if you are compiling the boost kernel
#DEFINES += SOFA_HAS_BOOST_KERNEL

DESTDIR = $$SOFA_DIR/bin
CONFIG += $$CONFIGPROJECTGUI
LIBS += $$SOFA_GUI_LIBS
LIBS += $$SOFA_LIBS

contains( DEFINES,SOFA_HAS_BOOST_KERNEL) { 
LIBS += -lsofaBoostKernel$$LIBSUFFIX
}

# The following is a workaround to get KDevelop to detect the name of the program to start
unix {
QMAKE_POST_LINK = ln -sf chainHybrid$$SUFFIX $$DESTDIR/chainHybrid-latest
}


SOURCES = Main.cpp 
HEADERS = 
