cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

message(STATUS "--------------------------------------------")
message(STATUS "-------- CONFIGURING SOFA FRAMEWORK --------")
message(STATUS "--------------------------------------------")
message(STATUS "")

include(cmake/pre.cmake)

# build flags

# disable partial inlining under gcc 4.6
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    EXECUTE_PROCESS( COMMAND g++ -dumpversion OUTPUT_VARIABLE GCXX_VERSION )
    if(${GCXX_VERSION} VERSION_EQUAL 4.6)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-partial-inlining")
    endif()
endif()

#   release/debug
# unless cmake is called with CMAKE_BUILD_TYPE=Debug, release build is forced
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

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
