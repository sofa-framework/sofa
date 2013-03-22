cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

include(cmake/pre.cmake)

message("")
message("--------------------------------------------")
message("-------- CONFIGURING SOFA FRAMEWORK --------")
message("--------------------------------------------")
message("")

# SOFA DEPENDENCIES
include(sofa-dependencies.cmake)

# PRINT REPORT
if(SOFA_ERROR_MESSAGE)
	message("Final report : ${SOFA_ERROR_MESSAGE}")
else()
	message("Final report : \n- Everything's OK !")
endif()

message("")
message("--------------------------------------------")
message("----- DONE CONFIGURING SOFA FRAMEWORK ------")
message("--------------------------------------------")
message("")

include(cmake/post.cmake)