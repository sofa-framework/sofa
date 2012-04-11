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


load(sofa/post)








