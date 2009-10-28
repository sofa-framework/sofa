# Target is a library:  csparse

SOFA_DIR = ../..
TEMPLATE = lib
TARGET = csparse

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $${CONFIGLIBRARIES}


DEFINES += use_namespace 

SOURCES = \    
        csparse.c \
        ldl.c \

HEADERS = \
        csparse.h \
        ldl.h \
        UFconfig.h \
