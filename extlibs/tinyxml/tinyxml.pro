# Target is a library:  tinyxml
load(sofa/pre)

TEMPLATE = lib
TARGET = tinyxml

DEFINES *= TIXML_USE_STL
CONFIGSTATIC = static

HEADERS += \
  tinystr.h \
  tinyxml.h

SOURCES += \
  tinystr.cpp \
  tinyxml.cpp \
  tinyxmlerror.cpp \
  tinyxmlparser.cpp

load(sofa/post)
