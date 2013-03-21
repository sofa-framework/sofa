cmake_minimum_required(VERSION 2.8)

# plugins


# extlibs
set(SOFA_MINI_BOOST "${SOFA_EXTLIBS_DIR}/miniBoost")
set(BOOST_PATH ${SOFA_MINI_BOOST} CACHE PATH "Use our minimal boost or type in the path of a full boost version")
if((NOT BOOST_PATH STREQUAL "") AND (NOT BOOST_PATH STREQUAL ${SOFA_MINI_BOOST}))
	set(SOFA_HAVE_BOOST 1)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_BOOST)
endif()

option(SOFA_HAVE_ZLIB "Use the ZLib library" ON)
if(SOFA_HAVE_ZLIB)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_ZLIB)
endif()

option(SOFA_HAVE_PNG "Use the LibPNG library" ON)
if(SOFA_HAVE_PNG)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_PNG)
endif()

option(SOFA_HAVE_GLEW "Use the GLEW library" ON)
if(SOFA_HAVE_GLEW)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_GLEW)
endif()

# developer convenience
option(DEV_SPLIT_HEADER_AND_CPP "Split headers and cpps in different filters" ON)

set(SOFA_HEADER_FILE_FILTER_NAME "")
set(SOFA_CPP_FILE_FILTER_NAME "")
if(DEV_SPLIT_HEADER_AND_CPP)
	set(SOFA_HEADER_FILE_FILTER_NAME	"Header Files")
	set(SOFA_CPP_FILE_FILTER_NAME		"Source Files")
endif()

# miscellaneous
option(MISC_DEV "Use the applications-dev folder" OFF)
option(MISC_UNIT_TEST "Build and use unit tests" OFF)

#
if(CMAKE_HOST_WIN32)
	list(APPEND GLOBAL_DEFINES "UNICODE")
endif()
