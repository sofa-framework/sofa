cmake_minimum_required(VERSION 2.8)

project("Qwt")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

	qwt.h
    qwt_abstract_scale_draw.h
    qwt_interval_symbol.h
    qwt_clipper.h
    qwt_color_map.h
    qwt_compat.h
    qwt_column_symbol.h
    qwt_interval.h
    qwt_dyngrid_layout.h
    qwt_global.h
    qwt_math.h
    qwt_magnifier.h
    qwt_null_paintdevice.h
    qwt_painter.h
    qwt_panner.h
    qwt_picker.h
    qwt_picker_machine.h
    qwt_point_3d.h
    qwt_point_polar.h
    qwt_round_scale_draw.h
    qwt_scale_div.h
    qwt_scale_draw.h
    qwt_scale_engine.h
    qwt_scale_map.h
    qwt_spline.h
    qwt_symbol.h
    qwt_system_clock.h
    qwt_text_engine.h
    qwt_text_label.h
    qwt_text.h
	)

set(SOURCE_FILES

	qwt_abstract_scale_draw.cpp
    qwt_interval_symbol.cpp
    qwt_clipper.cpp
    qwt_color_map.cpp
    qwt_column_symbol.cpp
    qwt_interval.cpp
    qwt_dyngrid_layout.cpp
    qwt_math.cpp
    qwt_magnifier.cpp
    qwt_panner.cpp
    qwt_null_paintdevice.cpp
    qwt_painter.cpp
    qwt_picker.cpp
    qwt_round_scale_draw.cpp
    qwt_scale_div.cpp
    qwt_scale_draw.cpp
    qwt_scale_map.cpp
    qwt_spline.cpp
    qwt_text_engine.cpp
    qwt_text_label.cpp
    qwt_text.cpp
    qwt_event_pattern.cpp
    qwt_picker_machine.cpp
    qwt_point_3d.cpp
    qwt_point_polar.cpp
    qwt_scale_engine.cpp
    qwt_symbol.cpp
    qwt_system_clock.cpp
	)
	
if(EXTERNAL_USE_QT4)
	find_package(Qt4 COMPONENTS qtopengl qt3support qtxml REQUIRED) 
	
	set(ENV{CONFIG} "qt")
	
	QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
else()
	find_package(Qt3 COMPONENTS qtopengl REQUIRED)
	
	set(ENV{CONFIG} "qt")
	
	QT3_WRAP_CPP(MOC_FILES ${HEADER_FILES})
endif()
set(QT_QMAKE_EXECUTABLE ${QT_QMAKE_EXECUTABLE} CACHE INTERNAL "QMake executable path")

include(${QT_USE_FILE})
include_directories(${QT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB QT_INCLUDE_SUBDIRS "${QT_INCLUDE_DIR}/*")
foreach(QT_INCLUDE_SUBDIR ${QT_INCLUDE_SUBDIRS})
	include_directories(${QT_INCLUDE_SUBDIR})
endforeach()

# make mocced header files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

# these header files do not need MOCcing
#list(APPEND HEADER_FILES "")

set(MOC_FILES_GROUP "Generated")
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES ${QT_LIBRARIES}) # need -lrt on linux (but not on mac)

include(${SOFA_CMAKE_DIR}/post.cmake)

