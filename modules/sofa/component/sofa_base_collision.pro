load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_base_collision

DEFINES += SOFA_BUILD_BASE_COLLISION

HEADERS += initBaseCollision.h \
           collision/BaseContactMapper.h \
           collision/DefaultPipeline.h \
           collision/Sphere.h \
           collision/SphereModel.h \
           collision/SphereModel.inl \
           collision/Cube.h \
           collision/CubeModel.h \
           collision/CubeModel.inl \
           collision/DiscreteIntersection.h \
           collision/DiscreteIntersection.inl \
           collision/BruteForceDetection.h \
           collision/DefaultContactManager.h \
           collision/Point.h \
           collision/PointModel.h \
           collision/Line.h \
           collision/LineModel.h \
           collision/Triangle.h \
           collision/TriangleModel.h \
           collision/TriangleModel.inl \
           collision/TetrahedronModel.h \
           collision/SpatialGridPointModel.h \
           collision/SphereTreeModel.h \
           collision/TriangleOctree.h \
           collision/TriangleOctreeModel.h \
           collision/RayModel.h \
           collision/LineLocalMinDistanceFilter.h \
           collision/LocalMinDistanceFilter.h \
           collision/PointLocalMinDistanceFilter.h \
           collision/TriangleLocalMinDistanceFilter.h \
           container/DistanceGrid.h \
           collision/Ray.h \
           collision/DistanceGridCollisionModel.h \
           collision/RayTriangleIntersection.h \
           collision/MinProximityIntersection.h \
           collision/NewProximityIntersection.h \
           collision/NewProximityIntersection.inl \

SOURCES += initBaseCollision.cpp \
           collision/BaseContactMapper.cpp \
           collision/DefaultPipeline.cpp \
           collision/SphereModel.cpp \
           collision/CubeModel.cpp \
           collision/DiscreteIntersection.cpp \
           collision/BruteForceDetection.cpp \
           collision/DefaultContactManager.cpp \
           collision/PointModel.cpp \
           collision/LineModel.cpp \
           collision/TriangleModel.cpp \
           collision/TetrahedronModel.cpp \
           collision/SpatialGridPointModel.cpp \
           collision/SphereTreeModel.cpp \
           collision/TriangleOctree.cpp \
           collision/TriangleOctreeModel.cpp \
           collision/RayModel.cpp \
           collision/LineLocalMinDistanceFilter.cpp \
           collision/LocalMinDistanceFilter.cpp \
           collision/PointLocalMinDistanceFilter.cpp \
           collision/TriangleLocalMinDistanceFilter.cpp \
           container/DistanceGrid.cpp \
           collision/DistanceGridCollisionModel.cpp \
           collision/RayTriangleIntersection.cpp \
           collision/MinProximityIntersection.cpp \
           collision/NewProximityIntersection.cpp \


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
