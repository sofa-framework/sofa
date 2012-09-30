load(sofa/pre)

TEMPLATE = lib

HEADERS += \
    Matrix_test.inl \

SOURCES += \
    Matrix_test.cpp \
 
LIBS += -lboost_unit_test_framework

load(sofa/post)


