cmake_minimum_required(VERSION 2.8.12)
project(miniFlowVR)

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
	include/flowvr/render/noise.h)

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
	src/librender/mesh_io_lwo.cpp)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

# The code must be relocatable if we want to link a shared library against it
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif()

#for preprocessor definition of MINI_FLOWVR
set(MINI_FLOWVR TRUE)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/MiniFlowVR.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/flowvr/MiniFlowVR.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/flowvr/MiniFlowVR.h
         DESTINATION "include/flowvr")

set_target_properties(miniFlowVR PROPERTIES DEBUG_POSTFIX "_d")
# set_target_properties(miniFlowVR PROPERTIES VERSION ${MINIFLOWVR_VERSION})
# set_target_properties(miniFlowVR PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(TARGETS miniFlowVR
        EXPORT MiniFlowVRTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

foreach(HEADER ${HEADER_FILES})
    string(REGEX MATCH "f(.*)[/\\]" DIR ${HEADER})
    install(FILES ${HEADER} DESTINATION "include/${DIR}")
endforeach(HEADER HEADERS)

include(CMakePackageConfigHelpers)

# MiniFlowVRConfigVersion.cmake file (same for build tree and intall)
# write_basic_package_version_file(MiniFlowVRConfigVersion.cmake
                                 # VERSION ${MINIFLOWVR_VERSION}
                                 # COMPATIBILITY AnyNewerVersion)

# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MiniFlowVRConfigVersion.cmake
        # DESTINATION lib/cmake/MiniFlowVR)


# MiniFlowVRConfig.cmake (build tree)
set(MINIFLOWVR_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(MiniFlowVRConfig.cmake.in
                              MiniFlowVRConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS MINIFLOWVR_INCLUDE_DIR)

# MiniFlowVRConfig.cmake (installed)
set(MINIFLOWVR_INCLUDE_DIR include)
configure_package_config_file(MiniFlowVRConfig.cmake.in
                              MiniFlowVRConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/MiniFlowVR
                              PATH_VARS MINIFLOWVR_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MiniFlowVRConfigInstall.cmake
        DESTINATION lib/cmake/MiniFlowVR
        RENAME MiniFlowVRConfig.cmake)

# MiniFlowVRTargets.cmake (installed)
install(EXPORT MiniFlowVRTargets
        DESTINATION lib/cmake/MiniFlowVR)
