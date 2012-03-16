load(sofa/pre)
defineAsPlugin(Compliant)

TARGET = Compliant

DEFINES += SOFA_BUILD_Compliant
DEFINES *= EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET


SOURCES = \
    initCompliant.cpp \
    BaseCompliance.cpp \
    Compliance.cpp \
    UniformCompliance.cpp \
    ComplianceSolver.cpp \

HEADERS = \
    initCompliant.h \
    BaseCompliance.h \
    Compliance.h \
    Compliance.inl \
    UniformCompliance.h \
    UniformCompliance.inl \
    ComplianceSolver.h \


README_FILE = Compliant.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"



load(sofa/eigen-unsupported) # sparse solvers

load(sofa/post)








