load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_implicit_ode_solver

DEFINES += SOFA_BUILD_IMPLICIT_ODE_SOLVER

HEADERS += odesolver/EulerImplicitSolver.h \
           odesolver/StaticSolver.h

SOURCES += odesolver/EulerImplicitSolver.cpp \
           odesolver/StaticSolver.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
