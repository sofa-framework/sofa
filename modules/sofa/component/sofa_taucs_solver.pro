load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_taucs_solver

DEFINES += SOFA_BUILD_TAUCS_SOLVER

HEADERS += initTaucsSolver.h \

SOURCES += initTaucsSolver.cpp \

contains(DEFINES,SOFA_HAVE_TAUCS){
win32{
  # BLAS
  LIBS *= -lblas_win32_MT
  # LAPACK
  LIBS *= -llapack_win32_MT
  # LAPACK
  }
  LIBS *= -lmetis$$LIBSUFFIX

HEADERS += linearsolver/SparseTAUCSSolver.h \
           linearsolver/IncompleteTAUCSSolver.h \

SOURCES += linearsolver/SparseTAUCSSolver.cpp \
           linearsolver/IncompleteTAUCSSolver.cpp \
}

contains(DEFINES,SOFA_EXTLIBS_TAUCS_MT){
    LIBS *= -lmetis$$LIBSUFFIX

    HEADERS += linearsolver/SparseTAUCSLLtSolver.h

    SOURCES += linearsolver/SparseTAUCSLLtSolver.cpp
}


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
