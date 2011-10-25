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
           collision/MinProximityIntersection.cpp \
           collision/NewProximityIntersection.cpp \


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
