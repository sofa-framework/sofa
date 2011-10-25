load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_eigen2_solver

DEFINES += SOFA_BUILD_EIGEN2_SOLVER

HEADERS += initEigen2Solver.h \
           linearsolver/EigenMatrixManipulator.h \
           linearsolver/SVDLinearSolver.h


SOURCES += initEigen2Solver.cpp \
           linearsolver/EigenMatrixManipulator.cpp \
           linearsolver/SVDLinearSolver.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
