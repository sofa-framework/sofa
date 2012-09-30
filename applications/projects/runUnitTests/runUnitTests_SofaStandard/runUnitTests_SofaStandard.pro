load(sofa/pre)

TEMPLATE = lib
TARGET  = runUnitTests_SofaStandard_test

SOURCES += \
    Matrix_test.cpp \
 
LIBS += -lboost_unit_test_framework

load(sofa/post)


