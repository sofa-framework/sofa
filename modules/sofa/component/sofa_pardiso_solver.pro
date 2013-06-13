load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_pardiso_solver

DEFINES += SOFA_BUILD_PARDISO_SOLVER

contains(DEFINES,SOFA_HAVE_PARDISO){

HEADERS += linearsolver/SparsePARDISOSolver.h \
	   initPardisoSolver.h

SOURCES += linearsolver/SparsePARDISOSolver.cpp \
	   initPardisoSolver.cpp

}

LIBS *= -lsofa_base_linear_solver -lpardiso412-GNU450-X86-64 -lblas -llapack

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
