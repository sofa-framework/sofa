load(sofa/pre)
defineAsPlugin(Flexible)

TARGET = Flexible

DEFINES += SOFA_BUILD_Flexible
DEFINES *= EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

SOURCES = initFlexible.cpp\
    BaseShapeFunction.cpp \
    ExtensionMapping.cpp \
    ShepardShapeFunction.cpp \
    LinearMapping.cpp \
    DeformationGradientTypes.cpp \
    StrainTypes.cpp \
    GreenStrainMapping.cpp \

HEADERS = initFlexible.h \
    BaseShapeFunction.h \
    ShepardShapeFunction.h \
    BaseJacobian.h \
    ExtensionMapping.h \
    ExtensionMapping.inl \
    DeformationGradientTypes.h \
    LinearMapping.h \
    LinearMapping.inl \
    LinearJacobianBlock.h \
    LinearJacobianBlock.inl \
    StrainTypes.h \
    GreenStrainMapping.h \
    GreenStrainMapping.inl \
    GreenStrainJacobianBlock.h \
    GreenStrainJacobianBlock.inl


README_FILE = Flexible.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR\"

load(sofa/post)
