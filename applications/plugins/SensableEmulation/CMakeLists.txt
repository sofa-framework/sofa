include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    #NewOmniDriverEmu.h 
	initSensableEmulation.h
    OmniDriverEmu.h
	pthread/pthread.h
	)

set(SOURCE_FILES

    #NewOmniDriverEmu.cpp 
    initSensableEmulation.cpp 
    OmniDriverEmu.cpp

	)

set(README_FILES

	PluginSensableEmulation.txt
	)

#IF(WIN32)
        #set(THREADS_USE_PTHREADS_WIN32 true)
#ENDIF()

find_package(Boost COMPONENTS thread REQUIRED)

#find_package(Threads REQUIRED)
#include_directories(${THREADS_PTHREADS_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_SENSABLEEMULATIONPLUGIN")
AddLinkerDependencies(SofaGuiQt SofaComponentGeneral SofaHaptics ${Boost_THREAD_LIBRARY})

include(${SOFA_CMAKE_DIR}/postProject.cmake)
