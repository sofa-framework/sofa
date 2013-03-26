cmake_minimum_required(VERSION 2.8)


cmake_policy(SET CMP0015 OLD)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

include(cmake/pre.cmake)

message(STATUS "--------------------------------------------")
message(STATUS "-------- CONFIGURING SOFA FRAMEWORK --------")
message(STATUS "--------------------------------------------")
message(STATUS "")

# SOFA DEPENDENCIES
include(sofa-dependencies.cmake)

# PRINT REPORT
if(SOFA_ERROR_MESSAGE)
	message(SEND_ERROR "Final report : ${SOFA_ERROR_MESSAGE}")
else()
	message(STATUS "Final report : \n- Everything's OK !")
endif()

message(STATUS "")
message(STATUS "--------------------------------------------")
message(STATUS "----- DONE CONFIGURING SOFA FRAMEWORK ------")
message(STATUS "--------------------------------------------")
message(STATUS "")

include(cmake/post.cmake)