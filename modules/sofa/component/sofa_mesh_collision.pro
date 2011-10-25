load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_mesh_collision

DEFINES += SOFA_BUILD_MESH_COLLISION

HEADERS += initMeshCollision.h \
           collision/MeshNewProximityIntersection.h \
           collision/MeshNewProximityIntersection.inl \
           collision/BarycentricPenalityContact.h \
           collision/BarycentricPenalityContact.inl \
           collision/BarycentricContactMapper.h \
           collision/BarycentricContactMapper.inl \
           collision/IdentityContactMapper.h \
           collision/IdentityContactMapper.inl \
		   collision/RigidContactMapper.h \
           collision/RigidContactMapper.inl \
           collision/SubsetContactMapper.h \
           collision/SubsetContactMapper.inl \
           collision/MeshDiscreteIntersection.h \
           collision/MeshDiscreteIntersection.inl \
           collision/MeshMinProximityIntersection.h \
           collision/Point.h \
           collision/PointModel.h \
           collision/Line.h \
           collision/LineModel.h \
           collision/Triangle.h \
           collision/TriangleModel.h \
           collision/TriangleModel.inl \
           collision/TriangleOctree.h \
           collision/TriangleOctreeModel.h \
           collision/RayTriangleIntersection.h \
           collision/PointLocalMinDistanceFilter.h \
           collision/LineLocalMinDistanceFilter.h \
           collision/TriangleLocalMinDistanceFilter.h \
           collision/LocalMinDistanceFilter.h \


SOURCES += initMeshCollision.cpp \
           collision/MeshNewProximityIntersection.cpp \
           collision/BarycentricPenalityContact.cpp \
           collision/BarycentricContactMapper.cpp \
           collision/IdentityContactMapper.cpp \
           collision/SubsetContactMapper.cpp \
	   collision/MeshDiscreteIntersection.cpp \
           collision/MeshMinProximityIntersection.cpp \
           collision/PointModel.cpp \
           collision/LineModel.cpp \
           collision/TriangleModel.cpp \
           collision/TriangleOctree.cpp \
           collision/TriangleOctreeModel.cpp \
           collision/RayTriangleIntersection.cpp \
           collision/PointLocalMinDistanceFilter.cpp \
           collision/LineLocalMinDistanceFilter.cpp \
           collision/TriangleLocalMinDistanceFilter.cpp \
           collision/LocalMinDistanceFilter.cpp \

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
