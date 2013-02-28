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
           collision/DiscreteIntersection.h \
           collision/DiscreteIntersection.inl \
           collision/BruteForceDetection.h \
           collision/DefaultContactManager.h \
           collision/MinProximityIntersection.h \
           collision/NewProximityIntersection.h \
           collision/NewProximityIntersection.inl \
           collision/CapsuleModel.h \
           collision/CapsuleModel.inl \
    collision/BaseProximityIntersection.h \
    collision/CapsuleIntTool.h \
    collision/OBBModel.inl \
    collision/OBBModel.h \
    collision/RigidSphereModel.h \
    collision/RigidSphereModel.inl \
    collision/OBBIntTool.h \
    collision/IntrOBBOBB.h \
    collision/IntrOBBOBB.inl \
    collision/IntrUtility3.h \
    collision/IntrUtility3.inl

SOURCES += initBaseCollision.cpp \
           collision/BaseContactMapper.cpp \
           collision/DefaultPipeline.cpp \
           collision/SphereModel.cpp \
           collision/CubeModel.cpp \
           collision/CapsuleModel.cpp \
           collision/DiscreteIntersection.cpp \
           collision/BruteForceDetection.cpp \
           collision/DefaultContactManager.cpp \
           collision/MinProximityIntersection.cpp \
           collision/NewProximityIntersection.cpp \
    collision/BaseProximityIntersection.cpp \    
    collision/CapsuleIntTool.cpp \
    collision/RigidSphereModel.cpp \
    collision/OBBModel.cpp \
    collision/OBBIntTool.cpp \
    collision/IntrOBBOBB.cpp \
    collision/IntrUtility3.cpp \
    collision/IntrCapsuleOBB.cpp





# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
