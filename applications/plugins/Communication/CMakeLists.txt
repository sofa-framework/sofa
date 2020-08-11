Cmake_minimum_required(VERSION 3.12)
project(Communication VERSION 1.0)

find_package(SofaFramework REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake" DESTINATION lib COMPONENT headers)

set(HEADER_FILES
    config.h
    components/CommunicationSubscriber.h
    components/CommunicationSubscriber.inl
    components/CommunicationCircularBuffer.h
    components/CommunicationCircularBuffer.inl
    components/ServerCommunication.h
    components/ServerCommunication.inl
    )
set(SOURCE_FILES
    initCommunication.cpp
    components/CommunicationSubscriber.cpp
    components/CommunicationCircularBuffer.cpp
    components/ServerCommunication.cpp
    )
set(EXTRA_FILES
    README.md
    )

set(PROTOCOL_LIBS "")

# ZMQ
find_package(ZeroMQ QUIET)
option(PLUGIN_COMMUNICATION_ZMQ "Make use of the ZMQ" ${ZeroMQ_FOUND})
if(PLUGIN_COMMUNICATION_ZMQ)
    sofa_find_package(ZeroMQ REQUIRED)
    message("Found ZMQ libraries")
    list(APPEND PROTOCOL_LIBS libzmq)
    sofa_install_libraries(TARGETS libzmq)

    list(APPEND HEADER_FILES
        components/ServerCommunicationZMQ.h
        components/ServerCommunicationZMQ.inl
        )
    list(APPEND SOURCE_FILES
        components/ServerCommunicationZMQ.cpp
        )
    list(APPEND EXTRA_FILES
        examples/ZMQ/primitives/Sender/senderRequest.scn
        examples/ZMQ/primitives/Sender/senderPublisher.scn
        examples/ZMQ/primitives/Receiver/receiverRequest.scn
        examples/ZMQ/primitives/Receiver/receiverPublisher.scn
        examples/ZMQ/matrix/Sender/senderRequest.scn
        examples/ZMQ/matrix/Sender/senderPublisher.scn
        examples/ZMQ/matrix/Receiver/receiverRequest.scn
        examples/ZMQ/matrix/Receiver/receiverPublisher.scn
        )
endif()

# Oscpack
find_package(Oscpack QUIET)
option(PLUGIN_COMMUNICATION_Oscpack "Make use of the Oscpack" ${OSCPACK_FOUND})
if(PLUGIN_COMMUNICATION_Oscpack)
    sofa_find_package(Oscpack REQUIRED)
    message("Found Oscpack libraries")
    include_directories(${Oscpack_INCLUDE_DIRS})
    list(APPEND PROTOCOL_LIBS ${Oscpack_LIBRARIES})
    if(WIN32)
       list(APPEND PROTOCOL_LIBS Ws2_32 winmm)
    endif()

    list(APPEND HEADER_FILES
        components/ServerCommunicationOSC.h
        components/ServerCommunicationOSC.inl
        )
    list(APPEND SOURCE_FILES
        components/ServerCommunicationOSC.cpp
        )
    list(APPEND EXTRA_FILES
        examples/OSC/primitives/Sender/sender.scn
        examples/OSC/primitives/Receiver/receiver.scn
        examples/OSC/matrix/Sender/sender.scn
        examples/OSC/matrix/Receiver/receiver.scn
        )
endif()

# VRPN
find_package(VRPN QUIET)
option(PLUGIN_COMMUNICATION_VRPN "Make use of the VRPN" ${VRPN_FOUND})
if(PLUGIN_COMMUNICATION_VRPN)
    sofa_find_package(VRPN REQUIRED)
    message("Found VRPN libraries")
    include_directories(${VRPN_INCLUDE_DIRS})
    list(APPEND PROTOCOL_LIBS ${VRPN_LIBRARIES})

    list(APPEND HEADER_FILES
        components/ServerCommunicationVRPN.h
        components/ServerCommunicationVRPN.inl
        )
    list(APPEND SOURCE_FILES
        components/ServerCommunicationVRPN.cpp
        )
    list(APPEND EXTRA_FILES
        examples/VRPN/primitives/Sender/Sender.scn
        examples/VRPN/primitives/Receiver/Receiver.scn
        examples/VRPN/matrix/Sender/Sender.scn
        examples/VRPN/matrix/Receiver/Receiver.scn
        )
endif()

if(WIN32)
    add_definitions(-D_WINSOCKAPI_)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaBaseLinearSolver SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} ${PROTOCOL_LIBS})

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(PLUGIN_COMMUNICATION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(PLUGIN_COMMUNICATION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(Communication_test)
endif()
