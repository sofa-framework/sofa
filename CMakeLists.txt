cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "Sofa")
project(${SOLUTION_NAME})

message(STATUS "--------------------------------------------")
message(STATUS "-------- CONFIGURING SOFA FRAMEWORK --------")
message(STATUS "--------------------------------------------")
message(STATUS "")


#   release/debug
# unless cmake is called with CMAKE_BUILD_TYPE=Debug, release build is forced
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

include(cmake/environment.cmake)

include(cmake/pre.cmake)

# BUILD FLAGS    
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    #message(STATUS "Building Debug")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
	message(STATUS "Building Release")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        EXECUTE_PROCESS( COMMAND g++ -dumpversion OUTPUT_VARIABLE GCXX_VERSION )
        set(CXX_OPTIMIZATION_FLAGS "-O2")
#        set(CXX_ARCH_FLAGS "-march='native'")
        set(CXX_STACKPROTECTOR_FLAGS "-fstack-protector --param=ssp-buffer-size=4")
        set(CXX_FORTIFYSOURCE_FLAGS  "-D_FORTIFY_SOURCE=2")
        set(CXX_WARNING_FLAGS "-Wall -W") 
        set(CMAKE_CXX_FLAGS_RELEASE 
           "${CXX_OPTIMIZATION_FLAGS} ${CXX_WARNING_FLAGS} ${CXX_ARCH_FLAGS} ${CXX_STACKPROTECTOR_FLAGS} ${CXX_FORTIFYSOURCE_FLAGS}" 
            CACHE STRING "Flags used by the compiler in Release builds" FORCE)
        # disable partial inlining under gcc 4.6
        if(${GCXX_VERSION} VERSION_EQUAL 4.6)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-partial-inlining")
        endif()
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    #messsage(STATUS "Building RelWithDebInfo")
elseif(CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
    #message(STATUS "Building MinSizeRel")
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
