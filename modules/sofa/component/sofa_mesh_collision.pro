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
           collision/SubsetContactMapper.h \
           collision/SubsetContactMapper.inl \
           collision/MeshDiscreteIntersection.h \
           collision/MeshDiscreteIntersection.inl \
           collision/MeshMinProximityIntersection.h \


SOURCES += initMeshCollision.cpp \
           collision/MeshNewProximityIntersection.cpp \
           collision/BarycentricPenalityContact.cpp \
           collision/BarycentricContactMapper.cpp \
           collision/IdentityContactMapper.cpp \
           collision/SubsetContactMapper.cpp \
	   collision/MeshDiscreteIntersection.cpp \
           collision/MeshMinProximityIntersection.cpp \


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
