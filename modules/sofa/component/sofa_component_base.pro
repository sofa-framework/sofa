load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_component_base

DEFINES += SOFA_BUILD_COMPONENT_BASE

HEADERS += initComponentBase.h

SOURCES += initComponentBase.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
