load(sofa/pre)
defineAsPlugin(ManifoldTopologies)

TARGET = ManifoldTopologies

TEMPLATE = lib

DEFINES += SOFA_BUILD_MANIFOLDTOPOLOGIES

SOURCES = initManifoldTopologies.cpp \
          ManifoldEdgeSetGeometryAlgorithms.h \
          ManifoldEdgeSetGeometryAlgorithms.inl \
          ManifoldEdgeSetTopologyAlgorithms.h \
          ManifoldEdgeSetTopologyAlgorithms.inl \
          ManifoldEdgeSetTopologyContainer.h \
          ManifoldEdgeSetTopologyModifier.h \
          ManifoldTriangleSetTopologyContainer.h \
          ManifoldTriangleSetTopologyModifier.h \
          ManifoldTriangleSetTopologyAlgorithms.h \
          ManifoldTriangleSetTopologyAlgorithms.inl \
          ManifoldTetrahedronSetTopologyContainer.h \
          ManifoldTopologyObject_double.h \
          ManifoldTopologyObject_float.h


HEADERS = ManifoldEdgeSetGeometryAlgorithms.cpp \
          ManifoldEdgeSetTopologyAlgorithms.cpp \
          ManifoldEdgeSetTopologyContainer.cpp \
          ManifoldEdgeSetTopologyModifier.cpp \
          ManifoldTriangleSetTopologyContainer.cpp \
          ManifoldTriangleSetTopologyModifier.cpp \
          ManifoldTriangleSetTopologyAlgorithms.cpp \
          ManifoldTetrahedronSetTopologyContainer.cpp


README_FILE = ManifoldTopologies.txt

#TODO: add an install target for README files

unix : QMAKE_POST_LINK = cp $$SRC_DIR/$$README_FILE $$LIB_DESTDIR 
win32 : QMAKE_POST_LINK = copy \"$$toWindowsPath($$SRC_DIR/$$README_FILE)\" \"$$LIB_DESTDIR"

load(sofa/post)
