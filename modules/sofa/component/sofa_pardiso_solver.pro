load(sofa/pre)

TEMPLATE = lib
TARGET = pardiso_solver

DEFINES += SOFA_BUILD_PARDISO_SOLVER

contains(DEFINES,SOFA_HAVE_PARDISO){

HEADERS += SparsePARDISOSolver.h

SOURCES += SparsePARDISOSolver.cpp

}

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
