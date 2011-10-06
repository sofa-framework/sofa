load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_collision

DEFINES += SOFA_BUILD_MISC_COLLISION

HEADERS += collision/TetrahedronModel.h \
           collision/SharpLineModel.h \
           collision/SpatialGridPointModel.h \
           collision/SphereTreeModel.h \
           collision/TriangleModelInRegularGrid.h \
           collision/TriangleOctree.h \
           collision/TriangleOctreeModel.h \
           collision/ContinuousIntersection.h \
           collision/ContinuousTriangleIntersection.h \
           collision/RigidContactMapper.h \
           collision/RigidContactMapper.inl \
           collision/TreeCollisionGroupManager.h \
           collision/BsplineModel.h \
           collision/BsplineModel.inl \
           collision/RuleBasedContactManager.h

SOURCES += collision/TetrahedronModel.cpp \
           collision/SharpLineModel.cpp \
           collision/SpatialGridPointModel.cpp \
           collision/SphereTreeModel.cpp \
           collision/TriangleModelInRegularGrid.cpp \
           collision/TriangleOctree.cpp \
           collision/TriangleOctreeModel.cpp \
           collision/ContinuousIntersection.cpp \
           collision/ContinuousTriangleIntersection.cpp \
           collision/RigidContactMapper.cpp \
           collision/TreeCollisionGroupManager.cpp \
           collision/BsplineModel.cpp \
           collision/RuleBasedContactManager.cpp

contains(DEFINES,SOFA_SMP){
HEADERS += collision/ParallelCollisionPipeline.h

SOURCES += collision/ParallelCollisionPipeline.cpp
}

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
