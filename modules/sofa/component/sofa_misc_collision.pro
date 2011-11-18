load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_collision

DEFINES += SOFA_BUILD_MISC_COLLISION

HEADERS += initMiscCollision.h \
           collision/TriangleModelInRegularGrid.h \
           collision/TreeCollisionGroupManager.h \
           collision/RuleBasedContactManager.h \
           collision/DefaultCollisionGroupManager.h \
           collision/SolverMerger.h \
           collision/TetrahedronDiscreteIntersection.h \
           collision/SpatialGridPointModel.h \
           collision/TetrahedronModel.h \

SOURCES += initMiscCollision.cpp \
		   collision/FrictionContact_DistanceGrid.cpp \
           collision/TriangleModelInRegularGrid.cpp \
           collision/TreeCollisionGroupManager.cpp \
           collision/RuleBasedContactManager.cpp \
           collision/DefaultCollisionGroupManager.cpp \
           collision/SolverMerger.cpp \
		   collision/TetrahedronDiscreteIntersection.cpp \
           collision/SpatialGridPointModel.cpp \
           collision/TetrahedronModel.cpp \
           collision/TetrahedronBarycentricPenalityContact.cpp \
           collision/TetrahedronRayContact.cpp \
           collision/TetrahedronFrictionContact.cpp \


contains(DEFINES,SOFA_SMP){
HEADERS += collision/ParallelCollisionPipeline.h

SOURCES += collision/ParallelCollisionPipeline.cpp
}

contains(DEFINES,SOFA_HAVE_EIGEN2){
SOURCES += collision/TetrahedronBarycentricDistanceLMConstraintContact.cpp \
		   collision/BarycentricDistanceLMConstraintContact_DistanceGrid.cpp
}

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
