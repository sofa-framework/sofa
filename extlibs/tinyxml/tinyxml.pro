# Target is a library:  sofatinyxml
load(sofa/pre)

TEMPLATE = lib
TARGET = tinyxml

DEFINES *= SOFA_BUILD_TINYXML

DEFINES *= TIXML_USE_STL
#CONFIGSTATIC = static

HEADERS += \
  tinystr.h \
  tinyxml.h

SOURCES += \
  tinystr.cpp \
  tinyxml.cpp \
  tinyxmlerror.cpp \
  tinyxmlparser.cpp

load(sofa/post)
