load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_implicit_ode_solver

DEFINES += SOFA_BUILD_IMPLICIT_ODE_SOLVER

HEADERS += initImplicitODESolver.h \
           odesolver/EulerImplicitSolver.h \
           odesolver/StaticSolver.h

SOURCES += initImplicitODESolver.cpp \
           odesolver/EulerImplicitSolver.cpp \
           odesolver/StaticSolver.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
