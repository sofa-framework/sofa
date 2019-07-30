cmake_minimum_required(VERSION 3.1)
project(miniFlowVR)

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
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/extlibs/MiniFlowVR>")

# The code must be relocatable if we want to link a shared library against it
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif()

include(${SOFA_KERNEL_SOURCE_DIR}/SofaFramework/SofaMacros.cmake)
sofa_generate_package(
    NAME MiniFlowVR
    VERSION 0.1
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "include"
    INCLUDE_INSTALL_DIR "extlibs/MiniFlowVR"
    )
