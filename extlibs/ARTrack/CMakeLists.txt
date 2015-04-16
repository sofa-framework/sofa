cmake_minimum_required(VERSION 2.8)
project(ARTrackLib)

set(HEADER_FILES dtracklib.h)
set(SOURCE_FILES mainTracker.cpp)

add_library(ARTrackLib ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_include_directories(ARTrackLib PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
set_target_properties(ARTrackLib PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

set(ARTRACK_COMPILER_FLAGS "-Duse_namespace")

# The code must be relocatable if we want to link a shared library against it
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(ARTRACK_COMPILER_FLAGS "${ARTRACK_COMPILER_FLAGS} -fPIC")
endif()

set_target_properties(ARTrackLib PROPERTIES COMPILE_FLAGS ${ARTRACK_COMPILER_FLAGS})
		  
install(TARGETS ARTrackLib
        EXPORT ARTrackLibTargets
        RUNTIME DESTINATION bin COMPONENT ARTrackLib_libraries
        LIBRARY DESTINATION lib COMPONENT ARTrackLib_libraries
        ARCHIVE DESTINATION lib COMPONENT ARTrackLib_libraries
        PUBLIC_HEADER DESTINATION include COMPONENT ARTrackLib_headers) 

include(CMakePackageConfigHelpers)

# ARTrackLibConfig.cmake (build tree)
set(ARTRACK_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
configure_package_config_file(ARTrackLibConfig.cmake.in 
                              ${CMAKE_BINARY_DIR}/ARTrackLib/ARTrackLibConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_BINARY_DIR}/ARTrackLib
                              PATH_VARS ARTRACK_INCLUDE_DIR)

# ARTrackLibConfig.cmake (installed)
set(ARTRACK_INCLUDE_DIR include)
configure_package_config_file(ARTrackLibConfig.cmake.in 
                              ${CMAKE_BINARY_DIR}/ARTrackLib/ARTrackLibConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/ARTrackLib
                              PATH_VARS ARTRACK_INCLUDE_DIR)

install(FILES ${CMAKE_BINARY_DIR}/ARTrackLib/ARTrackLibConfigInstall.cmake
        DESTINATION lib/cmake/ARTrackLib
        RENAME ARTrackLibConfig.cmake)
	   
# ARTrackLibTargets.cmake (installed)
install(EXPORT ARTrackLibTargets
        DESTINATION lib/cmake/ARTrackLib)
