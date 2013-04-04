cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

message(STATUS "--------------------------------------------")
message(STATUS "-------- CONFIGURING SOFA FRAMEWORK --------")
message(STATUS "--------------------------------------------")
message(STATUS "")

include(cmake/pre.cmake)

# SOFA DEPENDENCIES
include(sofa-dependencies.cmake)

# PRINT REPORT
if(SOFA_ERROR_MESSAGE)
	message(SEND_ERROR "Final report : ${SOFA_ERROR_MESSAGE}")
else()
	message(STATUS "Final report : \n- Everything's OK !")
endif()
message(STATUS "")

include(cmake/post.cmake)

message(STATUS "--------------------------------------------")
message(STATUS "----- DONE CONFIGURING SOFA FRAMEWORK ------")
message(STATUS "--------------------------------------------")
message(STATUS "")
