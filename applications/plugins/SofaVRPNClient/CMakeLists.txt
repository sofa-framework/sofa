cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    vrpnclient_config.h 
    VRPNDevice.h 
	VRPNAnalog.h 
	VRPNAnalog.inl 
	VRPNButton.h 
	VRPNTracker.h 
	VRPNTracker.inl 
	VRPNImager.h 
	VRPNImager.inl 
	WiimoteDriver.h 
	IRTracker.h 
	#ContactWarning.h 
	#ContactDisabler.h 
	ToolTracker.h 
	ToolTracker.inl 

	)

set(SOURCE_FILES

    initSofaVRPNClient.cpp 
	VRPNDevice.cpp 
	VRPNAnalog.cpp 
	VRPNButton.cpp 
	VRPNTracker.cpp 
	VRPNImager.cpp 
	WiimoteDriver.cpp 
	IRTracker.cpp 
	#ContactWarning.cpp 
 	#ContactDisabler.cpp 
	ToolTracker.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SOFAVRPNCLIENT")
set(LINKER_DEPENDENCIES SofaCore SofaSimulationCommon vrpn_quat vrpn_client)

include(${SOFA_CMAKE_DIR}/postProject.cmake)