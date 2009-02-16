# Target is a library:  tinyxml

SOFA_DIR = ../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = tinyxml$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES

HEADERS += \
  tinystr.h \
  tinyxml.h

SOURCES += \
  tinystr.cpp \
  tinyxml.cpp \
  tinyxmlerror.cpp \
  tinyxmlparser.cpp
