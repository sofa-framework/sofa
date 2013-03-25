cmake_minimum_required(VERSION 2.8)

# include dir
include_directories("${SOFA_INC_DIR}")
include_directories("${SOFA_FRAMEWORK_DIR}")
include_directories("${SOFA_MODULES_DIR}")
include_directories("${SOFA_APPLICATIONS_DIR}")

if(MISC_USE_DEV_PROJECTS)
	include_directories("${SOFA_APPLICATIONS_DEV_DIR}")
endif()

if(SOFA_BOOST_PATH)
	include_directories("${SOFA_BOOST_PATH}")
endif()

# lib dir
link_directories("${SOFA_LIB_DIR}")
link_directories("${SOFA_LIB_OS_DIR}")

# packages and libraries

## opengl / glew / glut
if(WIN32)
	set(OPENGL_LIBRARIES "opengl32")
	set(GLEW_LIBRARIES "glew32")
	set(GLUT_LIBRARIES "glut32")
	set(PNG_LIBRARIES "libpng")
else()
	find_library(OPENGL_LIBRARIES "opengl")
	find_library(GLEW_LIBRARIES "glew")
	find_library(GLUT_LIBRARIES "glut")
	find_library(PNG_LIBRARIES "png")
endif()
set(OPENGL_LIBRARIES ${OPENGL_LIBRARIES} CACHE INTERNAL "OpenGL Library")
set(GLEW_LIBRARIES ${GLEW_LIBRARIES} CACHE INTERNAL "GLEW Library")
set(GLUT_LIBRARIES ${GLUT_LIBRARIES} CACHE INTERNAL "GLUT Library")
set(PNG_LIBRARIES ${PNG_LIBRARIES} CACHE INTERNAL "PNG Library")

# target location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SOFA_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${SOFA_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${SOFA_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SOFA_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${SOFA_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${SOFA_BIN_DIR}")
