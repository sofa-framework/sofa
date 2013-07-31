################################################
# SOFA - ASCLEPIOS ## XICATH PLUGIN CONFIGURATION #
################################################
load(sofa/pre)
defineAsPlugin(sofa-asclepios)

TEMPLATE = lib

TARGET = PluginMultiThreading
#Debug:TARGET = PluginMultiThreadingDebug

win32{
DESTDIR = ..\..\..\..\..\lib}
unix{
DESTDIR = ../../../../../lib}


Release:OBJECTS_DIR = ../temp/obj/release
Debug:OBJECTS_DIR = ../temp/obj/debug

Release:MOC_DIR = ../temp/moc/release
Debug:MOC_DIR = ../temp/moc/debug


DEFINES += SOFA_MULTITHREADING_PLUGIN


#set configuration to dynamic library
contains (DEFINES, SOFA_QT4) {	
	CONFIG += qt 
	QT += opengl qt3support xml
}
else {
	CONFIG += qt
	QT += opengl
}

CONFIG += debug_and_release_target
		
			


SOURCES +=	initMultiThreading.cpp  \
			DataExchange.cpp \			
			#Observer.cpp \
			

HEADERS +=	initMultiThreading.h  \
			DataExchange.h \
			DataExchange.inl \			
			#Observer.h \

##----------------------------------------------------
##	other files (under conditions)
##----------------------------------------------------

## BOOST
contains(DEFINES, SOFA_HAVE_BOOST) {
	SOURCES +=	TaskSchedulerBoost.cpp \
				AnimationLoopParallelScheduler.cpp \							
				AnimationLoopTasks.cpp \
				Tasks.cpp \
				BeamLinearMapping_mt.cpp \
				##ParallelForTask.cpp \
			

	HEADERS +=	TaskSchedulerBoost.h \
				AnimationLoopParallelScheduler.h \
				AnimationLoopTasks.h \
				Tasks.h \
				Tasks.inl \				
				BeamLinearMapping_mt.h \
				BeamLinearMapping_mt.inl \
				BeamLinearMapping_tasks.inl \	
				##ParallelForTask.h \
}




load(sofa/post)
