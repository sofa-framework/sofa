cmake_minimum_required(VERSION 2.8)
project(OptiTrackNatNet)

set(OPTITRACKNATNET_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(OPTITRACKNATNET_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(OPTITRACKNATNET_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

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

find_package(SofaGeneral REQUIRED)
find_package(Boost QUIET COMPONENTS system REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_OPTITRACKNATNET")
target_link_libraries(${PROJECT_NAME} SofaCore SofaSimulationCommon SofaUserInteraction ${Boost_LIBRARIES})

# Seems to be required by boost::asio
if(UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

install(TARGETS ${PROJECT_NAME}
    COMPONENT OptiTrackNatNet_libraries
    EXPORT OptiTrackNatNetTargets
    RUNTIME DESTINATION "${OPTITRACKNATNET_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${OPTITRACKNATNET_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${OPTITRACKNATNET_INSTALL_LIB_DIR}"
    PUBLIC_HEADER DESTINATION "${OPTITRACKNATNET_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
