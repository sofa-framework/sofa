cmake_minimum_required(VERSION 2.8.12)
project(miniFlowVR)


set(MINIFLOWVR_BUILD_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(MINIFLOWVR_BUILD_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(MINIFLOWVR_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MINIFLOWVR_BUILD_CMAKE_DIR ${CMAKE_BINARY_DIR}/MiniFlowVR/)


# Offer the user the choice of overriding the installation directories
set(MINIFLOWVR_INSTALL_LIB_DIR     lib                   CACHE PATH "Installation directory for libraries")
set(MINIFLOWVR_INSTALL_BIN_DIR     bin                   CACHE PATH "Installation directory for executables")
set(MINIFLOWVR_INSTALL_INCLUDE_DIR include               CACHE PATH "Installation directory for header files")
set(MINIFLOWVR_INSTALL_CMAKE_DIR   ${MINIFLOWVR_INSTALL_LIB_DIR}/cmake/MiniFlowVR )

find_package(Boost REQUIRED)

set(HEADER_FILES

	include/ftl/cmdline.h 
	include/ftl/crc.h 
	include/ftl/fixed_array.h 
	include/ftl/mat.h 
	include/ftl/quat.h 
	include/ftl/rmath.h 
	include/ftl/type.h 
	include/ftl/vec.h 
	include/flowvr/render/bbox.h 
	include/flowvr/render/mesh.h 
	include/flowvr/render/mesh.inl 
	include/flowvr/render/noise.h
	)

set(SOURCE_FILES

	src/ftlm/cmdline.cpp 
	src/ftlm/crc.cpp 
	src/ftlm/mat.cpp 
	src/ftlm/quat.cpp 
	src/ftlm/type.cpp 
	src/librender/bbox.cpp 
	src/librender/mesh.cpp 
	src/librender/mesh_dist.cpp 
	src/librender/mesh_io_mesh.cpp 
	src/librender/mesh_io_obj.cpp 
	src/librender/mesh_io_off.cpp 
	src/librender/mesh_io_stl.cpp 
	src/librender/mesh_io_vtk.cpp 
	src/librender/mesh_io_lwo.cpp
	)
 
add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:${MINIFLOWVR_INSTALL_INCLUDE_DIR}>")

# The code must be relocatable if we want to link a shared library against it
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif()

#for preprocessor definition of MINI_FLOWVR  
set(MINI_FLOWVR TRUE)
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/MiniFlowVR.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/MiniFlowVR.h )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/MiniFlowVR.h
         DESTINATION "${MINIFLOWVR_INSTALL_INCLUDE_DIR}/miniFlowVR")

set_target_properties( miniFlowVR  PROPERTIES DEBUG_POSTFIX "_d" )
# set_target_properties( miniFlowVR  PROPERTIES VERSION ${MINIFLOWVR_VERSION} )
set_target_properties( miniFlowVR  PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS miniFlowVR
        EXPORT  MiniFlowVRTargets
        RUNTIME DESTINATION "${MINIFLOWVR_INSTALL_BIN_DIR}" 
        LIBRARY DESTINATION "${MINIFLOWVR_INSTALL_LIB_DIR}" 
        ARCHIVE DESTINATION "${MINIFLOWVR_INSTALL_LIB_DIR}" 
        PUBLIC_HEADER DESTINATION "${MINIFLOWVR_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" )
include(CMakePackageConfigHelpers)

# MiniFlowVRConfigVersion.cmake file (same for build tree and intall)
# write_basic_package_version_file(${MINIFLOWVR_BUILD_CMAKE_DIR}/MiniFlowVRConfigVersion.cmake
                                 # VERSION ${MINIFLOWVR_VERSION}
                                 # COMPATIBILITY AnyNewerVersion)
                                
# install(FILES ${MINIFLOWVR_BUILD_CMAKE_DIR}/MiniFlowVRConfigVersion.cmake
        # DESTINATION ${MINIFLOWVR_INSTALL_CMAKE_DIR})


# MiniFlowVRConfig.cmake (build tree)
set(MINIFLOWVR_INCLUDE_DIR "${MINIFLOWVR_BUILD_INCLUDE_DIR}")
configure_package_config_file(MiniFlowVRConfig.cmake.in 
                              ${MINIFLOWVR_BUILD_CMAKE_DIR}/MiniFlowVRConfig.cmake
                              INSTALL_DESTINATION ${MINIFLOWVR_BUILD_CMAKE_DIR}
                              PATH_VARS MINIFLOWVR_INCLUDE_DIR )

# MiniFlowVRConfig.cmake (installed)
set(MINIFLOWVR_INCLUDE_DIR "${MINIFLOWVR_INSTALL_INCLUDE_DIR}")
configure_package_config_file(MiniFlowVRConfig.cmake.in 
                              ${MINIFLOWVR_BUILD_CMAKE_DIR}/MiniFlowVRConfigInstall.cmake
                              INSTALL_DESTINATION ${MINIFLOWVR_INSTALL_CMAKE_DIR}
                              PATH_VARS MINIFLOWVR_INCLUDE_DIR )

install(FILES ${MINIFLOWVR_BUILD_CMAKE_DIR}/MiniFlowVRConfigInstall.cmake
        DESTINATION ${MINIFLOWVR_INSTALL_CMAKE_DIR}
        RENAME MiniFlowVRConfig.cmake)
	   
# MiniFlowVRTargets.cmake (installed)
install(EXPORT MiniFlowVRTargets
        DESTINATION ${MINIFLOWVR_INSTALL_CMAKE_DIR})
        

