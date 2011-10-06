load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_eigen2_solver

DEFINES += SOFA_BUILD_EIGEN2_SOLVER


contains(DEFINES,SOFA_HAVE_EIGEN2){
HEADERS += linearsolver/EigenMatrixManipulator.h \
           linearsolver/SVDLinearSolver.h


SOURCES += linearsolver/EigenMatrixManipulator.cpp \
           linearsolver/SVDLinearSolver.cpp
}

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
