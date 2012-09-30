load(sofa/pre)

TEMPLATE = lib

HEADERS += \
    initSofaStandard_test.h \
    Matrix_test.inl \

SOURCES += \
    initSofaStandard_test.cpp \
    Matrix_test.cpp \
 
LIBS += -lboost_unit_test_framework

load(sofa/post)


