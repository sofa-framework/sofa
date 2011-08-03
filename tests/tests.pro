SOFA_DIR = ..
TEMPLATE = app
TARGET = sofatests

include($${SOFA_DIR}/sofa.cfg)

DESTDIR = $$SOFA_DIR/bin

SOURCES = \
	main.cpp \
	framework/sofa/defaulttype/VecTypesTest.cpp \
	framework/sofa/helper/system/atomicTest.cpp \
	
	
contains(DEFINES, SOFA_HAVE_BOOST) {
	SOURCES += \
		framework/sofa/core/objectmodel/AspectPoolTest.cpp \ 
		framework/sofa/helper/system/thread/CircularQueueTest.cpp
}

contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--start-group
	LIBS += -Wl,--whole-archive
}
LIBS += $$SOFA_LIBS
contains(CONFIGSTATIC, static) {
	LIBS += -Wl,--no-whole-archive
	LIBS += -Wl,--end-group
}
