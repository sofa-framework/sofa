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

win32 {
	INCLUDEPATH += C:\H3D\External\include
	INCLUDEPATH += C:\H3D\External\include\pthread
	QMAKE_LIBDIR += C:\H3D\External\lib32
	INCLUDEPATH += C:\H3D\HAPI\include
	INCLUDEPATH += C:\H3D\H3DUtil\include
	QMAKE_LIBDIR += C:\H3D\lib
	
	win32-msvc2003 :               HSUFFIX = _vc7
	win32-msvc2005 :               HSUFFIX = _vc8
	win32-msvc2008 :               HSUFFIX = _vc9
	win32-msvc2010 :               HSUFFIX = _vc10
	CONFIG(debug, debug|release) : HSUFFIX += "_d"
	LIBS *= -lH3DUtil$$HSUFFIX -lHAPI$$HSUFFIX
} else {
	LIBS *= -lH3DUtil -lHAPI
}

load(sofa/post)
