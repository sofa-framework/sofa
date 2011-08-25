load(sofa/pre)

TEMPLATE = app
TARGET = sofatests

CONFIG += console

SOURCES = \
	main.cpp \
	framework/sofa/defaulttype/VecTypesTest.cpp \
	framework/sofa/helper/system/atomicTest.cpp \
	
	
contains(DEFINES, SOFA_HAVE_BOOST) {
	SOURCES += \
		framework/sofa/core/objectmodel/AspectPoolTest.cpp \ 
		framework/sofa/helper/system/thread/CircularQueueTest.cpp
}

load(sofa/post)
