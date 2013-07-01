include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    initOptiTrackNatNet.h 
	OptiTrackNatNetClient.h 
	OptiTrackNatNetDevice.h

	)

set(SOURCE_FILES

    initOptiTrackNatNet.cpp 
	OptiTrackNatNetClient.cpp 
	OptiTrackNatNetDevice.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_OPTITRACKNATNET")
AddLinkerDependencies(SofaCore SofaSimulationCommon SofaUserInteraction ${Boost_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)