load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_base_visual

DEFINES += SOFA_BUILD_BASE_VISUAL

HEADERS += visualmodel/BaseCamera.h \
           visualmodel/VisualModelImpl.h

SOURCES += visualmodel/BaseCamera.cpp \
           visualmodel/VisualModelImpl.cpp


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
