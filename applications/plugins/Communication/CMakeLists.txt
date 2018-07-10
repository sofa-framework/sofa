Cmake_minimum_required(VERSION 3.1)
project(Communication)

option(PLUGIN_COMMUNICATION_Oscpack "Make use of the Oscpack protocol" OFF)
option(PLUGIN_COMMUNICATION_ZMQ "Make use of the ZMQ protocol" OFF)
option(PLUGIN_COMMUNICATION_VRPN "Make use of the VRPN protocol" OFF)

set(COMMUNICATION_MAJOR_VERSION 1)
set(COMMUNICATION_MINOR_VERSION 0)
set(COMMUNICATION_VERSION ${COMMUNICATION_MAJOR_VERSION}.${COMMUNICATION_MINOR_VERSION})

set(HEADER_FILES
    config.h
    components/CommunicationSubscriber.h
    components/CommunicationSubscriber.inl
    components/communicationCircularBuffer.h
    components/communicationCircularBuffer.inl
    components/serverCommunication.h
    components/serverCommunication.inl

    )

set(SOURCE_FILES
    initCommunication.cpp
    components/CommunicationSubscriber.cpp
    components/communicationCircularBuffer.cpp
    components/serverCommunication.cpp
    )

set(EXTRA_FILES
    README.md
    examples/ZMQ/primitives/Sender/senderRequest.scn
    examples/ZMQ/primitives/Sender/senderPublisher.scn

    examples/ZMQ/primitives/Receiver/receiverRequest.scn
    examples/ZMQ/primitives/Receiver/receiverPublisher.scn

    examples/ZMQ/matrix/Sender/senderRequest.scn
    examples/ZMQ/matrix/Sender/senderPublisher.scn

    examples/ZMQ/matrix/Receiver/receiverRequest.scn
    examples/ZMQ/matrix/Receiver/receiverPublisher.scn

    examples/OSC/primitives/Sender/sender.scn

    examples/OSC/primitives/Receiver/receiver.scn

    examples/OSC/matrix/Sender/sender.scn

    examples/OSC/matrix/Receiver/receiver.scn

    examples/VRPN/primitives/Sender/Sender.scn
    examples/VRPN/primitives/Receiver/Receiver.scn

    examples/VRPN/matrix/Sender/Sender.scn
    examples/VRPN/matrix/Receiver/Receiver.scn
    )

find_package(SofaFramework REQUIRED)

#COMMUNICATION ZMQ
if(PLUGIN_COMMUNICATION_ZMQ)
    find_package(ZMQ REQUIRED)
    if (ZMQ_FOUND)
        message("Found ZMQ libraries")
        include_directories(${ZMQ_INCLUDE_DIRS})

        list(APPEND HEADER_FILES
            components/serverCommunicationZMQ.h
            components/serverCommunicationZMQ.inl
            )

        list(APPEND SOURCE_FILES
            components/serverCommunicationZMQ.cpp
            )
    else ()
        message("Can't find ZMQ !")
    endif ()

endif()

#COMMUNICATION OSC
if(PLUGIN_COMMUNICATION_Oscpack)
    find_package(Oscpack REQUIRED)
    if (Oscpack_FOUND)
        message("Found Oscpack libraries")
        include_directories(${Oscpack_INCLUDE_DIRS})

        list(APPEND HEADER_FILES
            components/serverCommunicationOSC.h
            components/serverCommunicationOSC.inl
            )

        list(APPEND SOURCE_FILES
            components/serverCommunicationOSC.cpp
            )
    else ()
        message("Can't find Oscpack !")
    endif ()
endif()

#COMMUNICATION VRPN
if(PLUGIN_COMMUNICATION_VRPN)
    find_package(VRPN REQUIRED)
    if (VRPN_FOUND)
        message("Found VRPN libraries")
        include_directories(${VRPN_INCLUDE_DIRS})

        list(APPEND HEADER_FILES
            components/serverCommunicationVRPN.h
            components/serverCommunicationVRPN.inl
            )

        list(APPEND SOURCE_FILES
            components/serverCommunicationVRPN.cpp
            )
    else ()
        message("Can't find VRPN !")
    endif ()

endif()

add_subdirectory(Communication_test)

IF(WIN32)
	add_definitions(-D_WINSOCKAPI_)
ENDIF(WIN32)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} ${Oscpack_LIBRARIES} ${ZMQ_LIBRARIES} ${VRPN_LIBRARIES} SofaCore SofaSimulationGraph)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_Communication")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
    COMPONENT Communication_libraries
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
