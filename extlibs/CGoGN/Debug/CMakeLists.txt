
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

file(
	GLOB
	shaders_srcD
	${CGoGN_ROOT_DIR}/include/Utils/*.frag
	${CGoGN_ROOT_DIR}/include/Utils/*.vert
	${CGoGN_ROOT_DIR}/include/Utils/*.geom	
	${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.frag
	${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.vert
	${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.geom
)


IF(WIN32)
	add_custom_target(shader_targetd shader_to_h ${shaders_srcD}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	SOURCES ${shaders_srcD} )
ELSE(WIN32)
	add_custom_target(shader_targetd shader_to_h ${shaders_srcD}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	SOURCES ${shaders_srcD} 
)
ENDIF(WIN32)


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

IF(WITH_QT)
	file(
		GLOB
		files_utils_qt
		${CGoGN_ROOT_DIR}/src/Utils/Qt/*.cpp
		${CGoGN_ROOT_DIR}/include/Utils/Qt/*.hpp
		${CGoGN_ROOT_DIR}/include/Utils/Qt/*.h
	)
	file(
		GLOB
		utils_qt_headers
		${CGoGN_ROOT_DIR}/include/Utils/Qt/qtgl.h
		${CGoGN_ROOT_DIR}/include/Utils/Qt/qtSimple.h
		${CGoGN_ROOT_DIR}/include/Utils/Qt/qtQGLV_glw.h
		${CGoGN_ROOT_DIR}/include/Utils/Qt/qtQGLV.h
		${CGoGN_ROOT_DIR}/include/Utils/Qt/qtpopup.h
		${CGoGN_ROOT_DIR}/include/Utils/Qt/qthistodraw.h 
		${CGoGN_ROOT_DIR}/include/Utils/Qt/qtcolorschooser.h
	)
		
	QT4_WRAP_CPP(UTILS_QT_HEADERS_MOC ${utils_qt_headers})
	SET (files_utils_withQt ${files_utils} ${files_utils_qt} ${UTILS_QT_HEADERS_MOC})
ENDIF (WITH_QT)


IF (ONELIB)
	IF(WITH_QT)
		add_library( cgognd ${files_topology} ${files_container} ${files_algo} ${files_utils_withQt}) 
	ELSE(WITH_QT)
		add_library( cgognd ${files_topology} ${files_container} ${files_algo} ${files_utils} )
	ENDIF (WITH_QT)
	add_dependencies(cgognd shader_targetd )
ELSE (ONELIB)
	add_library( topologyD	${files_topology})
	add_library( containerD	${files_container})
	add_library( algoD		${files_algo})
	IF(WITH_QT)
		add_library( utilsD ${files_utils_withQt} )
	ELSE(WITH_QT)
		add_library( utilsD ${files_utils} )
	ENDIF (WITH_QT)
	add_dependencies(utilsD shader_targetD )
ENDIF (ONELIB)
