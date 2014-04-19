cmake_minimum_required(VERSION 2.8)

## With this command you specify you have a project with target(s)
## after this, the default variable PROJECT_NAME is set with EmptyCmakePlugin
## so you can use ${PROJECT_NAME} to get the project name value
project(EmptyCmakePlugin)


## Try to auto get SOFA_DIR path 
## modifiable by user :
##  * using 'cmake-gui <sourceDirContainingACmakeLists.txt> ' and change the path for the variable
##  * using 'cmake -DSOFA_DIR=\path\to\sofa .. ' 
if(NOT SOFA_DIR)
    set(SOFA_DIR "$ENV{SOFA_DIR}" CACHE PATH "Sofa root directory")
endif()


## Add sofa/cmake dir to the SOFA modules ressource paths
## useful to find FindSOFA.cmake and other custom *.cmake files
list(APPEND CMAKE_MODULE_PATH "${SOFA_DIR}/cmake")


## find package : 
## if found, auto set the SOFA_FOUND variable to true (false otherwise)
find_package(SOFA)
if(SOFA_FOUND)
	include_directories(${SOFA_INCLUDE_DIRS}) # variable set by the FinSOFA.cmake
	add_definitions(-DHAS_SOFA) # allow preprocessing #DEFINE HAS_SOFA
else()
    message(FATAL_ERROR "Unable to find SOFA (see FindSOFA.cmake)")
endif()


## OUTPUT paths hierarchie in the build dir
## WIN32 is default variable in cmake like UNIX and MAC
if(WIN32)
  set(RUNTIME_OUTPUT_DIRECTORY bin)
  set(LIBRARY_OUTPUT_DIRECTORY bin)
  set(ARCHIVE_OUTPUT_DIRECTORY lib)
else()
  set(RUNTIME_OUTPUT_DIRECTORY bin)
  set(LIBRARY_OUTPUT_DIRECTORY lib)
  set(ARCHIVE_OUTPUT_DIRECTORY lib)
endif()
## CMAKE_BINARY_DIR is a default cmake variable set with the output build dir
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/${LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/${ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY    ${CMAKE_BINARY_DIR}/${RUNTIME_OUTPUT_DIRECTORY})


## LIST src
list(APPEND ${PROJECT_NAME}_HEADERS
    initEmptyCmakePlugin.h
)

list(APPEND ${PROJECT_NAME}_SOURCES
    initEmptyCmakePlugin.cpp
)

add_library(${PROJECT_NAME} MODULE 
        ${${PROJECT_NAME}_HEADERS}
        ${${PROJECT_NAME}_SOURCES}
)

target_link_libraries(${PROJECT_NAME} ${SOFA_LIBRARIES}) # variable set by the FinSOFA.cmake
