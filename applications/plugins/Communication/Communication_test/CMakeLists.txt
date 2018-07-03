cmake_minimum_required(VERSION 3.1)

project(Communication_test)

set(SOURCE_FILES
    Communication_test.cpp
)

#COMMUNICATION ZMQ
if(PLUGIN_COMMUNICATION_ZMQ)
    if (ZMQ_FOUND)
        list(APPEND SOURCE_FILES
	    Communication_testZMQ.cpp
	)
    endif ()
endif()

#COMMUNICATION OSC
if(PLUGIN_COMMUNICATION_Oscpack)
    if (Oscpack_FOUND)
        list(APPEND SOURCE_FILES
	    Communication_testOSC.cpp
	)
    endif ()
endif()

#COMMUNICATION VRPN
if(PLUGIN_COMMUNICATION_VRPN)
    if (VRPN_FOUND)
        list(APPEND SOURCE_FILES
	    Communication_testVRPN.cpp
	)
    endif ()
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest Communication ${Oscpack_LIBRARIES} ${ZMQ_LIBRARIES} ${VRPN_LIBRARIES})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
