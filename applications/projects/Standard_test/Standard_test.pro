load(sofa/pre)

TEMPLATE = app
TARGET = Standard_test

CONFIG += console

LIBS += -lgtest \
		-lgtest_main
		
INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/extlibs
INCLUDEPATH  += $$SOFA_INSTALL_INC_DIR/extlibs/gtest/include

QMAKE_LIBDIR += $$SOFA_INSTALL_INC_DIR/extlibs/gtest/lib

HEADERS += Sofa_test.h \
    Matrix_test.inl \
    Mapping_test.h \

SOURCES += \
    RigidMapping_test.cpp \
    Matrix_test.cpp \
    OBBTest.cpp \
    ProjectToPlaneConstraint_test.cpp \
    ProjectToLineConstraint_test.cpp \

win32 {
	QMAKE_CXXFLAGS_RELEASE += /MT
	QMAKE_CXXFLAGS_DEBUG += /MT
}

LIBS *= -L/opt/cuda/lib64

load(sofa/post)


