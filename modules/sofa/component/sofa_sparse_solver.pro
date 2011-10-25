load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_sparse_solver

DEFINES += SOFA_BUILD_SPARSE_SOLVER

HEADERS += initSparseSolver.h \
           linearsolver/PrecomputedLinearSolver.h \
           linearsolver/PrecomputedLinearSolver.inl \
           linearsolver/SparseCholeskySolver.h \
           linearsolver/SparseLUSolver.h \
           linearsolver/SparseLDLSolver.h \
           linearsolver/SparseLDLSolver.inl \
           linearsolver/SparseXXTSolver.h \
           linearsolver/SparseXXTSolver.inl

SOURCES += initSparseSolver.cpp \
           linearsolver/PrecomputedLinearSolver.cpp \
           linearsolver/SparseCholeskySolver.cpp \
           linearsolver/SparseLUSolver.cpp \
           linearsolver/SparseLDLSolver.cpp \
           linearsolver/SparseXXTSolver.cpp


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
