load(sofa/pre)
defineAsPlugin(Compliant)

TARGET = Compliant

DEFINES += SOFA_BUILD_Compliant
DEFINES += EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET


SOURCES = \
    ExtensionMapping.cpp \
    initCompliant.cpp \
    BaseCompliance.cpp \
    Compliance.cpp \
    UniformCompliance.cpp \
    ComplianceSolver.cpp \
    BaseShapeFunction.cpp \
    ShepardShapeFunction.cpp \
    LinearMapping.cpp

HEADERS = \
    ExtensionMapping.h \
    ExtensionMapping.inl \
    initCompliant.h \
    BaseCompliance.h \
    Compliance.h \
    Compliance.inl \
    UniformCompliance.h \
    UniformCompliance.inl \
    ComplianceSolver.h \
    BaseShapeFunction.h \
    ShepardShapeFunction.h \
    LinearMapping.h \
    LinearMapping.inl \
    BaseJacobian.h \
    LinearJacobianBlock.h \
    LinearJacobianBlock.inl

README_FILE = Compliant.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"



load(sofa/eigen-unsupported) # sparse solvers

load(sofa/post)








