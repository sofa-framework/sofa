load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc

DEFINES += SOFA_BUILD_MISC

HEADERS += initMisc.h \
           misc/MeshTetraStuffing.h

SOURCES += initMisc.cpp \
           misc/MeshTetraStuffing.cpp


contains(DEFINES,SOFA_SMP){
HEADERS += linearsolver/ParallelCGLinearSolver.h \
           linearsolver/ParallelCGLinearSolver.inl

SOURCES += linearsolver/ParallelCGLinearSolver.cpp
}

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
