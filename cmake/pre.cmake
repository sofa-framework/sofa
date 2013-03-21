cmake_minimum_required(VERSION 2.8)

message("> ${PROJECT_NAME} : Generating")

include(${CMAKE_CURRENT_LIST_DIR}/environment.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/functions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/options.cmake)

# include dir
include_directories("${SOFA_INC_DIR}")
include_directories("${SOFA_FRAMEWORK_DIR}")
include_directories("${SOFA_MODULES_DIR}")
include_directories("${SOFA_APPLICATIONS_DIR}")

if(MISC_DEV)
	include_directories("${SOFA_APPLICATIONS_DEV_DIR}")
endif()

if(BOOST_PATH)
	include_directories("${BOOST_PATH}")
endif()

# lib dir
link_directories("${SOFA_LIB_DIR}")
link_directories("${SOFA_LIB_OS_DIR}")

# packages and libraries
find_package(opengl)
if(CMAKE_HOST_WIN32)
	set(GLEW_LIB_NAME "glew32")
	set(GLUT_LIB_NAME "glut32")
else()
	set(GLEW_LIB_NAME "glew")
	set(GLUT_LIB_NAME "glut")
endif()
find_library(GLEW_LIBRARIES "${GLEW_LIB_NAME}" "${SOFA_LIB_OS_DIR}")
find_library(GLUT_LIBRARIES "${GLUT_LIB_NAME}" "${SOFA_LIB_OS_DIR}")