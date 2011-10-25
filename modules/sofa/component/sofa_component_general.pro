load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_component_general

DEFINES += SOFA_BUILD_COMPONENT_GENERAL

HEADERS += initComponentGeneral.h

SOURCES += initComponentGeneral.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
