load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_preconditioner

DEFINES += SOFA_BUILD_PRECONDITIONER

HEADERS += initPreconditioner.h \
           linearsolver/ShewchukPCGLinearSolver.h \
           linearsolver/PCGLinearSolver.h \
           linearsolver/JacobiPreconditioner.h \
           linearsolver/JacobiPreconditioner.inl \
           linearsolver/BlockJacobiPreconditioner.h \
           linearsolver/BlockJacobiPreconditioner.inl \
           linearsolver/SSORPreconditioner.h \
           linearsolver/WarpPreconditioner.h \
           linearsolver/WarpPreconditioner.inl \
           linearsolver/PrecomputedWarpPreconditioner.h \
           linearsolver/PrecomputedWarpPreconditioner.inl

SOURCES += initPreconditioner.cpp \
           linearsolver/ShewchukPCGLinearSolver.cpp \
           linearsolver/PCGLinearSolver.cpp \
           linearsolver/JacobiPreconditioner.cpp \
           linearsolver/BlockJacobiPreconditioner.cpp \
           linearsolver/SSORPreconditioner.cpp \
           linearsolver/WarpPreconditioner.cpp \
           linearsolver/PrecomputedWarpPreconditioner.cpp \

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
