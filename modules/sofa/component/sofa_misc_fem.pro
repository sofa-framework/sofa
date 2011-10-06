load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_fem

DEFINES += SOFA_BUILD_MISC_FEM

HEADERS += forcefield/FastTetrahedralCorotationalForceField.h \
           forcefield/FastTetrahedralCorotationalForceField.inl \
           forcefield/TetrahedralTensorMassForceField.h \
           forcefield/TetrahedralTensorMassForceField.inl 

SOURCES += forcefield/FastTetrahedralCorotationalForceField.cpp \
           forcefield/TetrahedralTensorMassForceField.cpp 

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
