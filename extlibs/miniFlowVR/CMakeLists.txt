cmake_minimum_required(VERSION 2.8)

project("miniFlowVR")

include(${SOFA_CMAKE_DIR}/pre.cmake)

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

include_directories("include")
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_COMPILER_DEFINES};MINI_FLOWVR")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

