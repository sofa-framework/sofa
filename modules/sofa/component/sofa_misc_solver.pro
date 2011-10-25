load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_solver

DEFINES += SOFA_BUILD_MISC_SOLVER

HEADERS += initMiscSolver.h \
           odesolver/DampVelocitySolver.h \
           odesolver/NewmarkImplicitSolver.h

SOURCES += initMiscSolver.cpp \
           odesolver/DampVelocitySolver.cpp \
           odesolver/NewmarkImplicitSolver.cpp 

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
