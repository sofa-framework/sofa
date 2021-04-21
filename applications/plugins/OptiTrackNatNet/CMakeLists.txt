cmake_minimum_required(VERSION 3.12)
project(OptiTrackNatNet)


set(HEADER_FILES
    OptiTrackNatNetClient.h
    OptiTrackNatNetDevice.h
    config.h
)

set(SOURCE_FILES
    OptiTrackNatNetClient.cpp
    OptiTrackNatNetDevice.cpp
    initOptiTrackNatNet.cpp
)

sofa_find_package(SofaSimulationCommon REQUIRED)
sofa_find_package(SofaUserInteraction REQUIRED)
sofa_find_package(Sofa.GL REQUIRED)
sofa_find_package(Boost QUIET COMPONENTS system regex date_time REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_OPTITRACKNATNET")
target_link_libraries(${PROJECT_NAME} SofaCore SofaSimulationCommon SofaUserInteraction Sofa.GL)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

# Seems to be required by boost::asio
if(UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

install(TARGETS ${PROJECT_NAME}
        COMPONENT OptiTrackNatNet_libraries
        EXPORT OptiTrackNatNetTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
