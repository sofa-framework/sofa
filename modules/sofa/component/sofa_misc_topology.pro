load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_topology

DEFINES += SOFA_BUILD_MISC_TOPOLOGY
DEFINES += POINT_DATA_VECTOR_ACCESS

HEADERS += topology/ManifoldEdgeSetGeometryAlgorithms.h \
           topology/ManifoldEdgeSetGeometryAlgorithms.inl \
           topology/ManifoldEdgeSetTopologyAlgorithms.h \
           topology/ManifoldEdgeSetTopologyAlgorithms.inl \
           topology/ManifoldEdgeSetTopologyContainer.h \
           topology/ManifoldEdgeSetTopologyModifier.h \
           topology/ManifoldTriangleSetTopologyContainer.h \
           topology/ManifoldTriangleSetTopologyModifier.h \
           topology/ManifoldTriangleSetTopologyAlgorithms.h \
           topology/ManifoldTriangleSetTopologyAlgorithms.inl \
           topology/ManifoldTetrahedronSetTopologyContainer.h \
           misc/TopologicalChangeProcessor.h

SOURCES += topology/ManifoldEdgeSetGeometryAlgorithms.cpp \
           topology/ManifoldEdgeSetTopologyAlgorithms.cpp \
           topology/ManifoldEdgeSetTopologyContainer.cpp \
           topology/ManifoldEdgeSetTopologyModifier.cpp \
           topology/ManifoldTriangleSetTopologyContainer.cpp \
           topology/ManifoldTriangleSetTopologyModifier.cpp \
           topology/ManifoldTriangleSetTopologyAlgorithms.cpp \
           topology/ManifoldTetrahedronSetTopologyContainer.cpp \
           misc/TopologicalChangeProcessor.cpp


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
