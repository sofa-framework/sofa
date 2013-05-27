load(sofa/pre)
defineAsPlugin(ExternalBehaviorModel)

TARGET = ExternalBehaviorModel

DEFINES += SOFA_BUILD_ExternalBehaviorModel


HEADERS = initExternalBehaviorModel.h \
      FEMGridBehaviorModel.h FEMGridBehaviorModel.inl \
	  

SOURCES = initExternalBehaviorModel.cpp \
      FEMGridBehaviorModel.cpp \


#win32 : QMAKE_CXXFLAGS += /bigobj

load(sofa/post)
