load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_engine

DEFINES += SOFA_BUILD_MISC_ENGINE

HEADERS += initMiscEngine.h \
           engine/Distances.h \
           engine/Distances.inl

SOURCES += initMiscEngine.cpp \
           engine/Distances.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
