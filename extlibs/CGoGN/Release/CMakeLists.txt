
# project(CGoGNRelease)

# SET(CMAKE_BUILD_TYPE Release)

# IF(WIN32)
# 	SET(LIBRARY_OUTPUT_PATH ${CGoGN_ROOT_DIR}/lib)
# ELSE(WIN32)
# 	SET(LIBRARY_OUTPUT_PATH ${CGoGN_ROOT_DIR}/lib/Release)
# 	ADD_DEFINITIONS(-DNOTOPOWARNING)
# ENDIF(WIN32)

include_directories(
	${CMAKE_CURRENT_BINARY_DIR}
	${CGoGN_ROOT_DIR}/include
	${CGoGN_EXT_INCLUDES}
)

file(
	GLOB_RECURSE
	files_topology
	${CGoGN_ROOT_DIR}/src/Topology/*.cpp
	${CGoGN_ROOT_DIR}/include/Topology/*.hpp
	${CGoGN_ROOT_DIR}/include/Topology/*.h
)

file(
	GLOB_RECURSE
	files_container
	${CGoGN_ROOT_DIR}/src/Container/*.cpp
	${CGoGN_ROOT_DIR}/include/Container/*.hpp
	${CGoGN_ROOT_DIR}/include/Container/*.h
)

file(
	GLOB_RECURSE
	files_algo
	${CGoGN_ROOT_DIR}/src/Algo/*.cpp
	${CGoGN_ROOT_DIR}/src/Algo/*.c
	${CGoGN_ROOT_DIR}/include/Algo/*.hpp
	${CGoGN_ROOT_DIR}/include/Algo/*.h
)



# IF(WIN32)
# 	add_custom_target(shader_target shader_to_h ${shaders_src}
# 	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# 	SOURCES ${shaders_src} )
# ELSE(WIN32)
# 	add_custom_target(shader_target shader_to_h ${shaders_src}
# 	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# 	SOURCES ${shaders_src} )
# ENDIF(WIN32)


file(
	GLOB		# WARNING NO MORE RECURSE TO AVOID TAKING QT FILES
	files_utils
	${CGoGN_ROOT_DIR}/src/Utils/*.cpp
	${CGoGN_ROOT_DIR}/include/Utils/*.hpp
	${CGoGN_ROOT_DIR}/include/Utils/*.h

	${CGoGN_ROOT_DIR}/src/Utils/Shaders/*.cpp
	${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.hpp
	${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.h
)



file(	GLOB_RECURSE
		files_geometry
		${CGoGN_ROOT_DIR}/include/Geometry/*.h
		${CGoGN_ROOT_DIR}/include/Geometry/*.hpp
)

file(	GLOB_RECURSE
		files_thirdParty
		${CGoGN_ROOT_DIR}/ThirdParty/include/*.h
		${CGoGN_ROOT_DIR}/ThirdParty/include/*.hpp
)


# IF (ONELIB)

	add_library(cgogn ${files_topology} ${files_container} ${files_algo} ${files_utils} )

	# add_dependencies(cgogn shader_target )


# 	get_property(dirs TARGET cgogn PROPERTY INCLUDE_DIRECTORIES)
# foreach(dir ${dirs})
#   message(STATUS "dir='${dir}'")
# endforeach()


# ELSE (ONELIB)
# 	add_library( container	${files_container})
# 	add_library( topology	${files_topology})
# 	add_library( algo		${files_algo})
# 	IF(WITH_QT)
# 		add_library( utils ${files_utils_withQt} )
# 	ELSE(WITH_QT)
# 		add_library( utils ${files_utils} )
# 	ENDIF (WITH_QT)
# 	add_custom_target( Geometry 	SOURCES ${files_geometry} )
# 	add_custom_target( ThirdParty 	SOURCES ${files_thirdParty} )
# 	add_dependencies( utils shader_target )
# ENDIF (ONELIB)