load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_solver

DEFINES += SOFA_BUILD_MISC_SOLVER

HEADERS += odesolver/DampVelocitySolver.h \
           odesolver/NewmarkImplicitSolver.h

SOURCES += odesolver/DampVelocitySolver.cpp \
           odesolver/NewmarkImplicitSolver.cpp 

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
