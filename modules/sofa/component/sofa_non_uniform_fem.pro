load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_non_uniform_fem

DEFINES += SOFA_BUILD_NON_UNIFORM_FEM

HEADERS += forcefield/NonUniformHexahedralFEMForceFieldAndMass.h \
           forcefield/NonUniformHexahedralFEMForceFieldAndMass.inl \
           forcefield/NonUniformHexahedronFEMForceFieldAndMass.h \
           forcefield/NonUniformHexahedronFEMForceFieldAndMass.inl \
           forcefield/NonUniformHexahedronFEMForceFieldDensity.h \
           forcefield/NonUniformHexahedronFEMForceFieldDensity.inl \
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

SOURCES += forcefield/NonUniformHexahedralFEMForceFieldAndMass.cpp \
           forcefield/NonUniformHexahedronFEMForceFieldAndMass.cpp \
           forcefield/NonUniformHexahedronFEMForceFieldDensity.cpp \
           topology/MultilevelHexahedronSetTopologyContainer.cpp \
           topology/DynamicSparseGridGeometryAlgorithms.cpp \
           topology/DynamicSparseGridTopologyAlgorithms.cpp \
           topology/DynamicSparseGridTopologyContainer.cpp \
           topology/DynamicSparseGridTopologyModifier.cpp \
           topology/SparseGridMultipleTopology.cpp \
           topology/SparseGridRamificationTopology.cpp \
           forcefield/HexahedronCompositeFEMForceFieldAndMass.cpp \
           mapping/HexahedronCompositeFEMMapping.cpp

contains(DEFINES,SOFA_DEV){
HEADERS += topology/MultilevelHexahedronSetGeometryAlgorithms.h \
           topology/MultilevelHexahedronSetGeometryAlgorithms.inl \
           topology/MultilevelHexahedronSetTopologyAlgorithms.h \
           topology/MultilevelHexahedronSetTopologyAlgorithms.inl \
           topology/MultilevelHexahedronSetTopologyModifier.h \

SOURCES += topology/MultilevelHexahedronSetGeometryAlgorithms.cpp \
           topology/MultilevelHexahedronSetTopologyAlgorithms.cpp \
           topology/MultilevelHexahedronSetTopologyModifier.cpp \

contains(DEFINES,SOFA_HAVE_EIGEN2){
HEADERS += forcefield/NonUniformHexahedralFEMForceFieldAndMassCorrected.h \
           forcefield/NonUniformHexahedralFEMForceFieldAndMassCorrected.inl

SOURCES += forcefield/NonUniformHexahedralFEMForceFieldAndMassCorrected.cpp
}

contains(DEFINES,SOFA_HAVE_GLEW) {
HEADERS += topology/Hexa2TriangleTopologicalMapping.h \
           topology/MultilevelHexaTopologicalMapping.h \
           topology/MultilevelHexa2TriangleTopologicalMapping.h \

SOURCES += topology/Hexa2TriangleTopologicalMapping.cpp \
           topology/MultilevelHexaTopologicalMapping.cpp \
           topology/MultilevelHexa2TriangleTopologicalMapping.cpp
}


}



# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
