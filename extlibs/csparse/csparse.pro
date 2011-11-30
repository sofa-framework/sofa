load(sofa/pre)

TEMPLATE = lib
TARGET = csparse

CONFIGSTATIC = static

DEFINES += use_namespace 

SOURCES = \    
        csparse.c \
        ldl.c \

HEADERS = \
        csparse.h \
        ldl.h \
        UFconfig.h \

load(sofa/post)
