Cmake_minimum_required(VERSION 3.1)
project(Communication)

option(PLUGIN_COMMUNICATION_Oscpack "Make use of the Oscpack protocol" OFF)
option(PLUGIN_COMMUNICATION_ZMQ "Make use of the ZMQ protocol" OFF)

set(COMMUNICATION_MAJOR_VERSION 1)
set(COMMUNICATION_MINOR_VERSION 0)
set(COMMUNICATION_VERSION ${COMMUNICATION_MAJOR_VERSION}.${COMMUNICATION_MINOR_VERSION})

add_subdirectory(Communication_test)

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
        include_directories(${OSCPACK_INCLUDE_DIRS})

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


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} ${Oscpack_LIBRARIES} ${ZMQ_LIBRARIES} SofaCore SofaSimulationGraph)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_Communication")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
    COMPONENT Communication_libraries
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
