load(sofa/pre)
defineAsPlugin(Compliant)

TARGET = Compliant

DEFINES += SOFA_BUILD_Compliant
DEFINES += EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
INCLUDEPATH += /usr/include/suitesparse  # for cholmod
INCLUDEPATH += /usr/include/superlu
LIBS += -lcholmod -lsuperlu


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


contains(DEFINES,SOFA_HAVE_EIGEN_UNSUPPORTED_AND_SPARSESUITE)
# ubuntu: apt-get install libsuitesparse-dev libeigen3-dev libsuperlu3 libsuperlu3-dev
{
INCLUDEPATH += $${SOFA_EIGEN_DIRECTORY}/unsupported  #   define this constant in your sofa-local.prf file, e.g. SOFA_EIGEN_DIRECTORY=/home/ffaure/local/sofa-dev/trunk/Sofa/extlibs/eigen-3.0.5
INCLUDEPATH += /usr/include/suitesparse  # for cholmod
INCLUDEPATH += /usr/include/superlu
LIBS += -lcholmod -lsuperlu

DEFINES += EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
}


load(sofa/post)








