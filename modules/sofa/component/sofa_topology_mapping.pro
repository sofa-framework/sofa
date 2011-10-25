load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_topology_mapping

DEFINES += SOFA_BUILD_TOPOLOGY_MAPPING

HEADERS += initTopologyMapping.h \
           mapping/Mesh2PointMechanicalMapping.h \
           mapping/Mesh2PointMechanicalMapping.inl \
           mapping/SimpleTesselatedTetraMechanicalMapping.h \
           mapping/SimpleTesselatedTetraMechanicalMapping.inl \
           topology/CenterPointTopologicalMapping.h \
           topology/Edge2QuadTopologicalMapping.h \
           topology/Hexa2QuadTopologicalMapping.h \
           topology/Hexa2TetraTopologicalMapping.h \
           topology/Mesh2PointTopologicalMapping.h  \
           topology/Quad2TriangleTopologicalMapping.h \
           topology/SimpleTesselatedHexaTopologicalMapping.h \
           topology/SimpleTesselatedTetraTopologicalMapping.h \
           topology/Tetra2TriangleTopologicalMapping.h \
           topology/Triangle2EdgeTopologicalMapping.h


SOURCES += initTopologyMapping.cpp \
           mapping/Mesh2PointMechanicalMapping.cpp \
           mapping/SimpleTesselatedTetraMechanicalMapping.cpp \
           topology/CenterPointTopologicalMapping.cpp \
           topology/Edge2QuadTopologicalMapping.cpp \
           topology/Hexa2QuadTopologicalMapping.cpp \
           topology/Hexa2TetraTopologicalMapping.cpp \
           topology/Mesh2PointTopologicalMapping.cpp  \
           topology/Quad2TriangleTopologicalMapping.cpp \
           topology/SimpleTesselatedHexaTopologicalMapping.cpp \
           topology/SimpleTesselatedTetraTopologicalMapping.cpp \
           topology/Tetra2TriangleTopologicalMapping.cpp \
           topology/Triangle2EdgeTopologicalMapping.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
