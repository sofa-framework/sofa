load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_graph_component

DEFINES += SOFA_BUILD_GRAPH_COMPONENT

HEADERS += initGraphComponent.h \
           configurationsetting/AddFrameButtonSetting.h \
           configurationsetting/AttachBodyButtonSetting.h \
           configurationsetting/BackgroundSetting.h \
           configurationsetting/FixPickedParticleButtonSetting.h \
           configurationsetting/MouseButtonSetting.h \
           configurationsetting/SofaDefaultPathSetting.h \
           configurationsetting/StatsSetting.h \
           configurationsetting/ViewerSetting.h \
           contextobject/Gravity.h \
           misc/PauseAnimation.h \
           misc/PauseAnimationOnEvent.h \
           misc/RequiredPlugin.h

SOURCES += initGraphComponent.cpp \
           configurationsetting/AddFrameButtonSetting.cpp \
           configurationsetting/AttachBodyButtonSetting.cpp \
           configurationsetting/BackgroundSetting.cpp \
           configurationsetting/FixPickedParticleButtonSetting.cpp \
           configurationsetting/MouseButtonSetting.cpp \
           configurationsetting/SofaDefaultPathSetting.cpp \
           configurationsetting/StatsSetting.cpp \
           configurationsetting/ViewerSetting.cpp \
           contextobject/Gravity.cpp \
           misc/PauseAnimation.cpp \
           misc/PauseAnimationOnEvent.cpp \
           misc/RequiredPlugin.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
