include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    OptiTrackNatNetClient.h
    OptiTrackNatNetDevice.h
    initOptiTrackNatNet.h
)

set(SOURCE_FILES
    OptiTrackNatNetClient.cpp
    OptiTrackNatNetDevice.cpp
    initOptiTrackNatNet.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_OPTITRACKNATNET")
AddLinkerDependencies(SofaCore SofaSimulationCommon SofaUserInteraction ${Boost_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)