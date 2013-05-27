# This module defines the following variables:
#  OPENGL_INCLUDE_DIRS - include directories for OPENGL
#  OPENGL_LIBRARIES - libraries to link against OPENGL
#  OPENGL_FOUND - true if OPENGL has been found and can be used

if (NOT DEFINED OPENGL_FOUND)

	if(WIN32)
		set(OPENGL_LIBRARIES "opengl32")
		set(OPENGL_FOUND CACHE INTERNAL TRUE)
	endif(WIN32)
	
	if(PS3)
		set(OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/PSGL/RSX/ultra-opt/libPSGL.a")
		list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/PSGL/RSX/ultra-opt/libPSGLU.a")
		list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libgcm_cmd.a")
		list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libgcm_pm.a")
		list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libgcm_sys_stub.a")
		list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libsysutil_avconf_ext_stub.a")
		list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libresc_stub.a")
		list (APPEND OPENGL_LIBRARIES "$(SCE_PS3_ROOT)/target/ppu/lib/libsysutil_stub.a")
		set(OPENGL_FOUND CACHE INTERNAL TRUE)
		message( "OPENGL PS3 LIBS" ${OPENGL_LIBRARIES} )
	endif(PS3)

	if(UNIX)
		if(APPLE)
			find_path(OPENGL_INCLUDE_DIR OpenGL/gl.h DOC "Include for OpenGL on OSX")
			#find_library(OPENGL_gl_LIBRARY OpenGL DOC "OpenGL lib for OSX")
			#find_library(OPENGL_glu_LIBRARY AGL DOC "AGL lib for OSX")
			find_library(OPENGL_LIBRARIES OpenGL DOC "OpenGL lib for OSX")
		else(APPLE)
			find_path(OPENGL_INCLUDE_DIR GL/gl.h)
			#find_library(OPENGL_gl_LIBRARY NAMES opengl32 OpenGL GL MesaGL)
			#find_library(OPENGL_glu_LIBRARY NAMES opengl32 OpenGL GL MesaGL)
			find_library(OPENGL_LIBRARIES NAMES opengl32 OpenGL GL MesaGL)
		endif(APPLE)

		if(EXISTS ${OPENGL_LIBRARIES})
			#message(STATUS "OPENGL Lib found: ${OPENGL_LIBRARIES} ${OPENGL_INCLUDE_DIR}")
			set(OPENGL_FOUND CACHE INTERNAL TRUE)
		else()
			message(FATAL_ERROR "OPENGL NOT FOUND ${OPENGL_LIBRARIES} ${OPENGL_INCLUDE_DIR}")
			set(OPENGL_FOUND FALSE)
		endif()

	endif(UNIX)



endif(NOT DEFINED OPENGL_FOUND)

mark_as_advanced(OPENGL_INCLUDE_DIR OPENGL_LIBRARIES)
