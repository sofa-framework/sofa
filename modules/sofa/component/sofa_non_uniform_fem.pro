load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_non_uniform_fem

DEFINES += SOFA_BUILD_NON_UNIFORM_FEM

HEADERS += initNonUniformFEM.h \
           forcefield/NonUniformHexahedralFEMForceFieldAndMass.h \
           forcefield/NonUniformHexahedralFEMForceFieldAndMass.inl \
           forcefield/NonUniformHexahedronFEMForceFieldAndMass.h \
           forcefield/NonUniformHexahedronFEMForceFieldAndMass.inl \
##           forcefield/NonUniformHexahedronFEMForceFieldDensity.h \
##           forcefield/NonUniformHexahedronFEMForceFieldDensity.inl \
           topology/DynamicSparseGridGeometryAlgorithms.h \
           topology/DynamicSparseGridGeometryAlgorithms.inl \
           topology/DynamicSparseGridTopologyAlgorithms.h \
           topology/DynamicSparseGridTopologyAlgorithms.inl \
           topology/DynamicSparseGridTopologyContainer.h \
           topology/DynamicSparseGridTopologyModifier.h \
           topology/MultilevelHexahedronSetTopologyContainer.h \
           topology/SparseGridMultipleTopology.h \
           topology/SparseGridRamificationTopology.h \
           forcefield/HexahedronCompositeFEMForceFieldAndMass.h \
           forcefield/HexahedronCompositeFEMForceFieldAndMass.inl \
           mapping/HexahedronCompositeFEMMapping.h \
           mapping/HexahedronCompositeFEMMapping.inl

SOURCES += initNonUniformFEM.cpp \
           forcefield/NonUniformHexahedralFEMForceFieldAndMass.cpp \
           forcefield/NonUniformHexahedronFEMForceFieldAndMass.cpp \
##           forcefield/NonUniformHexahedronFEMForceFieldDensity.cpp \
           topology/MultilevelHexahedronSetTopologyContainer.cpp \
           topology/DynamicSparseGridGeometryAlgorithms.cpp \
           topology/DynamicSparseGridTopologyAlgorithms.cpp \
           topology/DynamicSparseGridTopologyContainer.cpp \
           topology/DynamicSparseGridTopologyModifier.cpp \
           topology/SparseGridMultipleTopology.cpp \
           topology/SparseGridRamificationTopology.cpp \
           forcefield/HexahedronCompositeFEMForceFieldAndMass.cpp \
           mapping/HexahedronCompositeFEMMapping.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
