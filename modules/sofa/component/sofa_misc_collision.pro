load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_collision

DEFINES += SOFA_BUILD_MISC_COLLISION

HEADERS += initMiscCollision.h \
           collision/TriangleModelInRegularGrid.h \
           collision/RigidContactMapper.h \
           collision/RigidContactMapper.inl \
           collision/TreeCollisionGroupManager.h \
           collision/RuleBasedContactManager.h \
           collision/DefaultCollisionGroupManager.h \
           collision/SolverMerger.h \
           collision/TetrahedronDiscreteIntersection.h \
           collision/RayDiscreteIntersection.h \
           collision/RayDiscreteIntersection.inl \
           collision/RigidDistanceGridDiscreteIntersection.h \
           collision/RigidDistanceGridDiscreteIntersection.inl \
           collision/FFDDistanceGridDiscreteIntersection.h \
           collision/FFDDistanceGridDiscreteIntersection.inl \
           collision/RayNewProximityIntersection.h \


SOURCES += initMiscCollision.cpp \
	   collision/BarycentricPenalityContact_DistanceGrid.cpp \
	   collision/FrictionContact_DistanceGrid.cpp \
           collision/TriangleModelInRegularGrid.cpp \
           collision/RigidContactMapper.cpp \
           collision/TreeCollisionGroupManager.cpp \
           collision/RuleBasedContactManager.cpp \
           collision/DefaultCollisionGroupManager.cpp \
           collision/SolverMerger.cpp \
	   collision/TetrahedronDiscreteIntersection.cpp \
	   collision/RayDiscreteIntersection.cpp \
	   collision/RigidDistanceGridDiscreteIntersection.cpp \
	   collision/FFDDistanceGridDiscreteIntersection.cpp \
           collision/RayNewProximityIntersection.cpp \


contains(DEFINES,SOFA_SMP){
HEADERS += collision/ParallelCollisionPipeline.h

SOURCES += collision/ParallelCollisionPipeline.cpp
}

contains(DEFINES,SOFA_HAVE_EIGEN2){
SOURCES += collision/BarycentricDistanceLMConstraintContact_DistanceGrid.cpp
}

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
