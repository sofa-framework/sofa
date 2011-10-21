load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_base_topology

DEFINES += SOFA_BUILD_BASE_TOPOLOGY

HEADERS += initBaseTopology.h \
           topology/CommonAlgorithms.h \
           topology/CubeTopology.h \
           topology/CylinderGridTopology.h \
#           topology/EdgeData.h \
#           topology/EdgeData.inl \
           topology/EdgeSetGeometryAlgorithms.h \
           topology/EdgeSetGeometryAlgorithms.inl \
           topology/EdgeSetTopologyAlgorithms.h \
           topology/EdgeSetTopologyAlgorithms.inl \
           topology/EdgeSetTopologyContainer.h \
           topology/EdgeSetTopologyModifier.h \
           topology/EdgeSetTopologyEngine.h \
           topology/EdgeSetTopologyEngine.inl \
           topology/EdgeSubsetData.h \
           topology/EdgeSubsetData.inl \
           topology/GridTopology.h \
#           topology/HexahedronData.h \
#           topology/HexahedronData.inl \
           topology/HexahedronSetGeometryAlgorithms.h \
           topology/HexahedronSetGeometryAlgorithms.inl \
           topology/HexahedronSetTopologyAlgorithms.h \
           topology/HexahedronSetTopologyAlgorithms.inl \
           topology/HexahedronSetTopologyContainer.h \
           topology/HexahedronSetTopologyModifier.h \
           topology/HexahedronSetTopologyEngine.h \
           topology/HexahedronSetTopologyEngine.inl \
           topology/MeshTopology.h \
#           topology/PointData.h \
#          topology/PointData.inl \
           topology/PointSetTopologyEngine.h \
           topology/PointSetTopologyEngine.inl \
           topology/PointSetGeometryAlgorithms.h \
           topology/PointSetGeometryAlgorithms.inl \
           topology/PointSetTopologyAlgorithms.h \
           topology/PointSetTopologyAlgorithms.inl \
           topology/PointSetTopologyContainer.h \
           topology/PointSetTopologyModifier.h \
           topology/PointSubsetData.h \
           topology/PointSubsetData.inl \
#           topology/QuadData.h \
#           topology/QuadData.inl \
           topology/QuadSetGeometryAlgorithms.h \
           topology/QuadSetGeometryAlgorithms.inl \
           topology/QuadSetTopologyAlgorithms.h \
           topology/QuadSetTopologyAlgorithms.inl \
           topology/QuadSetTopologyContainer.h \
           topology/QuadSetTopologyModifier.h \
           topology/QuadSetTopologyEngine.h \
           topology/QuadSetTopologyEngine.inl \
           topology/RegularGridTopology.h \
           topology/SparseGridTopology.h \
#           topology/TetrahedronData.h \
#           topology/TetrahedronData.inl \
           topology/TetrahedronSetGeometryAlgorithms.h \
           topology/TetrahedronSetGeometryAlgorithms.inl \
           topology/TetrahedronSetTopologyAlgorithms.h \
           topology/TetrahedronSetTopologyAlgorithms.inl \
           topology/TetrahedronSetTopologyContainer.h \
           topology/TetrahedronSetTopologyModifier.h \
           topology/TetrahedronSetTopologyEngine.h \
           topology/TetrahedronSetTopologyEngine.inl \
           topology/TopologyChangedEvent.h \
#           topology/TriangleData.h \
#           topology/TriangleData.inl \
           topology/TriangleSetGeometryAlgorithms.h \
           topology/TriangleSetGeometryAlgorithms.inl \
           topology/TriangleSetTopologyAlgorithms.h \
           topology/TriangleSetTopologyAlgorithms.inl \
           topology/TriangleSetTopologyContainer.h \
           topology/TriangleSetTopologyModifier.h \
           topology/TriangleSetTopologyEngine.h \
           topology/TriangleSetTopologyEngine.inl \
           topology/TriangleSubsetData.h \
           topology/TriangleSubsetData.inl \
           topology/QuadSubsetData.h \
           topology/QuadSubsetData.inl \
           topology/TetrahedronSubsetData.h \
           topology/TetrahedronSubsetData.inl \
           topology/HexahedronSubsetData.h \
           topology/HexahedronSubsetData.inl \
           topology/TopologyData.h \
           topology/TopologyData.inl \
#           topology/TopologySparseDataHandler.h \
#           topology/TopologySparseDataHandler.inl \
#           topology/TopologySparseData.h \
#           topology/TopologySparseData.inl \
#           topology/TopologySubsetData.h \
#           topology/TopologySubsetData.inl \
           topology/TopologyDataHandler.h \
	   topology/TopologyDataHandler.inl \ 
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
           topology/PointSubsetData.cpp \
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

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
