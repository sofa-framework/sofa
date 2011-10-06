load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_dense_solver

DEFINES += SOFA_BUILD_DENSE_SOLVER

HEADERS += linearsolver/LULinearSolver.h \
           linearsolver/NewMatVector.h \
           linearsolver/NewMatMatrix.h \
           linearsolver/NewMatCGLinearSolver.h \
           linearsolver/NewMatCholeskySolver.h

SOURCES += linearsolver/LULinearSolver.cpp \
           linearsolver/NewMatCGLinearSolver.cpp \
           linearsolver/NewMatCholeskySolver.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
