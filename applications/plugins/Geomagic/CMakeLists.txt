cmake_minimum_required(VERSION 2.8.12)
project(Geomagic)

set(HEADER_FILES
      src/GeomagicDriver.h
      src/initPlugin.h
)

set(SOURCE_FILES
      src/GeomagicDriver.cpp
      src/initPlugin.cpp
)

set(SCENES_FILES
      scenes/DemoGeomagic.scn
)


find_package(OpenHaptics)

if( NOT OPENHAPTICS_FOUND )
	message(WARNING "OpenHaptics library not found, GeomagicDriver cannot compile")
else()
	include_directories( ${OPENHAPTICS_INCLUDE_DIR})

	add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${SCENES_FILES} ${README_FILES})
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DPLUGIN_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")
	target_link_libraries(${PROJECT_NAME} SofaHelper SofaOpenglVisual SofaUserInteraction ${OPENHAPTICS_LIBRARIES})
endif( NOT OPENHAPTICS_FOUND )