load(sofa/pre)
defineAsPlugin(Compliant)

TARGET = Compliant

DEFINES += SOFA_BUILD_Compliant
DEFINES += SOFA_HAVE_EIGEN_UNSUPPORTED_AND_CHOLMOD


SOURCES = \
    initCompliant.cpp \
    CompliantAttachButtonSetting.cpp \
    CompliantAttachPerformer.cpp \
    UniformCompliance.cpp \
    DiagonalCompliance.cpp \
    ComplianceSolver.cpp \
    MinresSolver.cpp \

HEADERS = \
    initCompliant.h \
    CompliantAttachButtonSetting.h \
    CompliantAttachPerformer.h \
    CompliantAttachPerformer.inl \
    UniformCompliance.h \
    UniformCompliance.inl \
    ComplianceSolver.h \
    MinresSolver.h \


README_FILE = Compliant.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

win32 {
SOFA_CHOLMOD_PATH = ../../../../extlibs/SuiteSparse/cholmod
INCLUDEPATH += $$SOFA_CHOLMOD_PATH/Include
QMAKE_LIBDIR += $$SOFA_CHOLMOD_PATH/Lib
LIBS += -lamd
LIBS += -llapack
LIBS += -lblas
LIBS += -lcamd
LIBS += -lcolamd
LIBS += -lcholmod

QMAKE_POST_LINK = copy \"$$toWindowsPath($$SOFA_CHOLMOD_PATH/Lib/libblas.dll)\" \"../../../../bin/libblas.dll\" && copy \"$$toWindowsPath($$SOFA_CHOLMOD_PATH/Lib/liblapack.dll)\" \"../../../../bin/liblapack.dll\" && copy \"$$toWindowsPath($$SOFA_CHOLMOD_PATH/Lib/libgcc_s_dw2-1.dll)\" \"../../../../bin/libgcc_s_dw2-1.dll\" && copy \"$$toWindowsPath($$SOFA_CHOLMOD_PATH/Lib/libgfortran-3.dll)\" \"../../../../bin/libgfortran-3.dll\"
}

unix {
INCLUDEPATH += /usr/include/suitesparse/
LIBS += -lamd
LIBS += -llapack
LIBS += -lblas
LIBS += -lcamd
LIBS += -lcolamd
LIBS += -lcholmod
}

load(sofa/post)








