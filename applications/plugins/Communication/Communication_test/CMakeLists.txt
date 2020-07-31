cmake_minimum_required(VERSION 3.1)

project(Communication_test)

set(SOURCE_FILES
    Communication_test.cpp
)

# ZMQ
if(PLUGIN_COMMUNICATION_ZMQ AND ZMQ_FOUND)
    list(APPEND SOURCE_FILES
        Communication_testZMQ.cpp
    )
endif()

# OSC
if(PLUGIN_COMMUNICATION_Oscpack AND Oscpack_FOUND)
    list(APPEND SOURCE_FILES
        Communication_testOSC.cpp
    )
endif()

# VRPN
if(PLUGIN_COMMUNICATION_VRPN AND VRPN_FOUND)
    list(APPEND SOURCE_FILES
        Communication_testVRPN.cpp
    )
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest Communication)
target_link_libraries(${PROJECT_NAME} ${PROTOCOL_LIBS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
