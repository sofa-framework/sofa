# release/debug

## unless cmake is called with CMAKE_BUILD_TYPE=Debug, release build is forced
if(NOT CMAKE_BUILD_TYPE)
	#message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

# build flags
if(CMAKE_BUILD_TYPE MATCHES "Debug")
	#message(STATUS "Building Debug")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
	#message(STATUS "Building Release")
	if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
		execute_process(COMMAND g++ -dumpversion OUTPUT_VARIABLE GCXX_VERSION)
		set(CXX_OPTIMIZATION_FLAGS "-O2")
		#set(CXX_ARCH_FLAGS "-march='native'")
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

set(compilerDefines ${GLOBAL_COMPILER_DEFINES})
if(WIN32)
	list(APPEND compilerDefines "UNICODE")
endif()

# NDEBUG preprocessor macro
if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND compilerDefines "NDEBUG")
endif()

if(PS3) 
    list(APPEND compilerDefines "SOFA_FLOAT") 
    list(APPEND compilerDefines "SOFA_NO_EXTERN_TEMPLATE") 
endif() 
 	
# SOFA_DEBUG preprocessor macro
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND compilerDefines "SOFA_DEBUG")
endif()

set(GLOBAL_COMPILER_DEFINES ${compilerDefines} CACHE INTERNAL "Global Compiler Defines" FORCE)
