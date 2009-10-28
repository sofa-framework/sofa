# Target is a library:  tinyxml

SOFA_DIR = ../..
TEMPLATE = lib
TARGET = tinyxml

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGLIBRARIES

HEADERS += \
  tinystr.h \
  tinyxml.h

SOURCES += \
  tinystr.cpp \
  tinyxml.cpp \
  tinyxmlerror.cpp \
  tinyxmlparser.cpp
