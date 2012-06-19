load(sofa/pre)
defineAsPlugin(SofaHAPI)

TARGET = SofaHAPI

DEFINES += SOFA_BUILD_SOFAHAPI

HEADERS = SofaHAPI.h \
          SofaHAPIHapticsDevice.h \
          SofaHAPIForceFeedbackEffect.h

SOURCES = initSofaHAPI.cpp \
          SofaHAPIHapticsDevice.cpp \
          SofaHAPIForceFeedbackEffect.cpp

LIBS *= -lH3DUtil -lHAPI

load(sofa/post)
