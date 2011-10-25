load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_base_animation_loop

DEFINES += SOFA_BUILD_BASE_ANIMATION_LOOP

HEADERS += initBaseAnimationLoop.h \
           animationloop/MultiStepAnimationLoop.h \
           animationloop/MultiTagAnimationLoop.h

SOURCES += initBaseAnimationLoop.cpp \
           animationloop/MultiStepAnimationLoop.cpp \
           animationloop/MultiTagAnimationLoop.cpp


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
