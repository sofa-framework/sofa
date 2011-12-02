load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_user_interaction

DEFINES += SOFA_BUILD_USER_INTERACTION

HEADERS += initUserInteraction.h \
           collision/RayTraceDetection.h \
           collision/RayContact.h \
           collision/ComponentMouseInteraction.h \
           collision/ComponentMouseInteraction.inl \
           collision/MouseInteractor.h \
           collision/MouseInteractor.inl \
           collision/AttachBodyPerformer.h \
           collision/AttachBodyPerformer.inl \
           collision/FixParticlePerformer.h \
           collision/FixParticlePerformer.inl \
           collision/InteractionPerformer.h \
           collision/SuturePointPerformer.h \
           collision/SuturePointPerformer.inl \
           controller/ArticulatedHierarchyController.h \
           controller/ArticulatedHierarchyBVHController.h \
           controller/Controller.h \
           controller/EdgeSetController.h \
           controller/EdgeSetController.inl \
           controller/MechanicalStateController.h \
           controller/MechanicalStateController.inl \
           collision/Ray.h \
           collision/RayModel.h \
           collision/RayDiscreteIntersection.h \
           collision/RayDiscreteIntersection.inl \
           collision/RayNewProximityIntersection.h \
           collision/RemovePrimitivePerformer.h \
           collision/RemovePrimitivePerformer.inl \
           collision/InciseAlongPathPerformer.h \
           collision/TopologicalChangeManager.h \


SOURCES += initUserInteraction.cpp \
           collision/RayTraceDetection.cpp \
           collision/RayContact.cpp \
           collision/ComponentMouseInteraction.cpp \
           collision/MouseInteractor.cpp \
           collision/AttachBodyPerformer.cpp \
           collision/FixParticlePerformer.cpp \
           collision/InteractionPerformer.cpp \
           collision/SuturePointPerformer.cpp \
           controller/ArticulatedHierarchyController.cpp \
           controller/ArticulatedHierarchyBVHController.cpp \
           controller/Controller.cpp \
           controller/EdgeSetController.cpp \
           controller/MechanicalStateController.cpp \
           collision/RayModel.cpp \
           collision/RayDiscreteIntersection.cpp \
           collision/RayNewProximityIntersection.cpp \
           collision/RemovePrimitivePerformer.cpp \
           collision/InciseAlongPathPerformer.cpp \
           collision/TopologicalChangeManager.cpp \


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
