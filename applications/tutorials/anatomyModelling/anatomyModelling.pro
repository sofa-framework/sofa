load(sofa/pre)

TEMPLATE = app
TARGET = anatomyModelling

INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs/
INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/extlibs/CImg
INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/applications/plugins/
INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/applications/plugins/image
INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/applications/plugins/Flexible
INCLUDEPATH += $$SOFA_INSTALL_INC_DIR/applications-dev/plugins/SohusimDev

SOURCES = anatomyModelling.cpp
HEADERS =

load(sofa/post)
