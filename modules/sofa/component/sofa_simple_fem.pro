load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_simple_fem

DEFINES += SOFA_BUILD_SIMPLE_FEM

HEADERS += initSimpleFEM.h \
           forcefield/BeamFEMForceField.h \
           forcefield/BeamFEMForceField.inl \
           forcefield/HexahedralFEMForceField.h \
           forcefield/HexahedralFEMForceField.inl \
           forcefield/HexahedralFEMForceFieldAndMass.h \
           forcefield/HexahedralFEMForceFieldAndMass.inl \
           forcefield/HexahedronFEMForceField.h \
           forcefield/HexahedronFEMForceField.inl \
           forcefield/HexahedronFEMForceFieldAndMass.h \
           forcefield/HexahedronFEMForceFieldAndMass.inl \
           forcefield/TetrahedralCorotationalFEMForceField.h \
           forcefield/TetrahedralCorotationalFEMForceField.inl \
           forcefield/TetrahedronFEMForceField.h \
           forcefield/TetrahedronFEMForceField.inl \
           forcefield/TriangularAnisotropicFEMForceField.h \
           forcefield/TriangularAnisotropicFEMForceField.inl \
           forcefield/TriangleFEMForceField.h \
           forcefield/TriangleFEMForceField.inl \
           forcefield/TriangularFEMForceField.h \
           forcefield/TriangularFEMForceField.inl \
           container/PoissonContainer.h \
           container/StiffnessContainer.h \
           container/RadiusContainer.h \
           container/LengthContainer.h \

SOURCES += initSimpleFEM.cpp \
           forcefield/BeamFEMForceField.cpp \
           forcefield/HexahedralFEMForceField.cpp \
           forcefield/HexahedralFEMForceFieldAndMass.cpp \
           forcefield/HexahedronFEMForceField.cpp \
           forcefield/HexahedronFEMForceFieldAndMass.cpp \
           forcefield/TetrahedralCorotationalFEMForceField.cpp \
           forcefield/TetrahedronFEMForceField.cpp \
           forcefield/TriangularAnisotropicFEMForceField.cpp \
           forcefield/TriangleFEMForceField.cpp \
           forcefield/TriangularFEMForceField.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
