load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_explicit_ode_solver

DEFINES += SOFA_BUILD_EXPLICIT_ODE_SOLVER

HEADERS += initExplicitODESolver.h \
           odesolver/CentralDifferenceSolver.h \
           odesolver/EulerSolver.h \
           odesolver/RungeKutta2Solver.h \
           odesolver/RungeKutta4Solver.h


SOURCES += initExplicitODESolver.cpp \
           odesolver/CentralDifferenceSolver.cpp \
           odesolver/EulerSolver.cpp \
           odesolver/RungeKutta2Solver.cpp \
           odesolver/RungeKutta4Solver.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
