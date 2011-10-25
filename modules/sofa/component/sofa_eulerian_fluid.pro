load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_eulerian_fluid

DEFINES += SOFA_BUILD_EULERIAN_FLUID

HEADERS += initEulerianFluid.h \
           behaviormodel/eulerianfluid/Fluid2D.h \
           behaviormodel/eulerianfluid/Fluid3D.h \
           behaviormodel/eulerianfluid/Grid2D.h \
           behaviormodel/eulerianfluid/Grid3D.h

SOURCES += initEulerianFluid.cpp \
           behaviormodel/eulerianfluid/Fluid2D.cpp \
           behaviormodel/eulerianfluid/Fluid3D.cpp \
           behaviormodel/eulerianfluid/Grid2D.cpp \
           behaviormodel/eulerianfluid/Grid3D.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
