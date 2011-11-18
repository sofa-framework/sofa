load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_base_topology

DEFINES += SOFA_BUILD_BASE_TOPOLOGY

HEADERS += initBaseTopology.h \
           topology/CommonAlgorithms.h \
           topology/CubeTopology.h \
           topology/CylinderGridTopology.h \
           topology/EdgeSetGeometryAlgorithms.h \
           topology/EdgeSetGeometryAlgorithms.inl \
           topology/EdgeSetTopologyAlgorithms.h \
           topology/EdgeSetTopologyAlgorithms.inl \
           topology/EdgeSetTopologyContainer.h \
           topology/EdgeSetTopologyModifier.h \
           topology/GridTopology.h \
           topology/HexahedronSetGeometryAlgorithms.h \
           topology/HexahedronSetGeometryAlgorithms.inl \
           topology/HexahedronSetTopologyAlgorithms.h \
           topology/HexahedronSetTopologyAlgorithms.inl \
           topology/HexahedronSetTopologyContainer.h \
           topology/HexahedronSetTopologyModifier.h \
           topology/MeshTopology.h \
           topology/PointSetGeometryAlgorithms.h \
           topology/PointSetGeometryAlgorithms.inl \
           topology/PointSetTopologyAlgorithms.h \
           topology/PointSetTopologyAlgorithms.inl \
           topology/PointSetTopologyContainer.h \
           topology/PointSetTopologyModifier.h \
           topology/QuadSetGeometryAlgorithms.h \
           topology/QuadSetGeometryAlgorithms.inl \
           topology/QuadSetTopologyAlgorithms.h \
           topology/QuadSetTopologyAlgorithms.inl \
           topology/QuadSetTopologyContainer.h \
           topology/QuadSetTopologyModifier.h \
           topology/RegularGridTopology.h \
           topology/SparseGridTopology.h \
           topology/TetrahedronSetGeometryAlgorithms.h \
           topology/TetrahedronSetGeometryAlgorithms.inl \
           topology/TetrahedronSetTopologyAlgorithms.h \
           topology/TetrahedronSetTopologyAlgorithms.inl \
           topology/TetrahedronSetTopologyContainer.h \
           topology/TetrahedronSetTopologyModifier.h \
           topology/TriangleSetGeometryAlgorithms.h \
           topology/TriangleSetGeometryAlgorithms.inl \
           topology/TriangleSetTopologyAlgorithms.h \
           topology/TriangleSetTopologyAlgorithms.inl \
           topology/TriangleSetTopologyContainer.h \
           topology/TriangleSetTopologyModifier.h \
           topology/TopologyData.h \
           topology/TopologyData.inl \
           topology/TopologyDataHandler.h \
	   topology/TopologyDataHandler.inl \ 
           topology/TopologySparseDataHandler.h \
           topology/TopologySparseDataHandler.inl \
           topology/TopologySparseData.h \
           topology/TopologySparseData.inl \
           topology/TopologySubsetData.h \
           topology/TopologySubsetData.inl \
           topology/TopologySubsetDataHandler.h \
	   topology/TopologySubsetDataHandler.inl \ 
	   topology/TopologyEngine.h \
	   topology/TopologyEngine.inl

SOURCES += initBaseTopology.cpp \
           topology/CubeTopology.cpp \
           topology/CylinderGridTopology.cpp \
           topology/EdgeSetGeometryAlgorithms.cpp \
           topology/EdgeSetTopologyAlgorithms.cpp \
           topology/EdgeSetTopologyContainer.cpp \
           topology/EdgeSetTopologyModifier.cpp \
           topology/GridTopology.cpp \
           topology/HexahedronSetGeometryAlgorithms.cpp \
           topology/HexahedronSetTopologyAlgorithms.cpp \
           topology/HexahedronSetTopologyContainer.cpp \
           topology/HexahedronSetTopologyModifier.cpp \
           topology/MeshTopology.cpp \
           topology/PointSetGeometryAlgorithms.cpp \
           topology/PointSetTopologyAlgorithms.cpp \
           topology/PointSetTopologyContainer.cpp \
           topology/PointSetTopologyModifier.cpp \
           topology/QuadSetGeometryAlgorithms.cpp \
           topology/QuadSetTopologyAlgorithms.cpp \
           topology/QuadSetTopologyContainer.cpp \
           topology/QuadSetTopologyModifier.cpp \
           topology/RegularGridTopology.cpp \
           topology/SparseGridTopology.cpp \
           topology/TetrahedronSetGeometryAlgorithms.cpp \
           topology/TetrahedronSetTopologyAlgorithms.cpp \
           topology/TetrahedronSetTopologyContainer.cpp \
           topology/TetrahedronSetTopologyModifier.cpp \
           topology/TriangleSetGeometryAlgorithms.cpp \
           topology/TriangleSetTopologyAlgorithms.cpp \
           topology/TriangleSetTopologyContainer.cpp \
           topology/TriangleSetTopologyModifier.cpp


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
