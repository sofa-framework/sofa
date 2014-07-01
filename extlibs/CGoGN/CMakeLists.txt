include(${SOFA_CMAKE_DIR}/preProject.cmake)
# cmake_minimum_required(VERSION 2.8)

# project(CGoGN)

#SET ( CMAKE_VERBOSE_MAKEFILE 1 )

SET(CGoGN_ROOT_DIR ${SOFA_EXTLIBS_DIR}/CGoGN )

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
				 	  ${CGoGN_ROOT_DIR}/cmake_modules/ )

# for shared or not shared
# SET ( BUILD_SHARED_LIBS OFF CACHE BOOL "if used all library are build as shared type (.so/.dylib/.dll)" )

# SET ( WITH_ASSIMP ON CACHE BOOL "build CGoGN with Assimp" )
# SET ( WITH_ZINRI ON CACHE BOOL "build CGoGN with Zinri lib" )
# SET ( WITH_QT ON CACHE BOOL "build CGoGN with Qt lib" )
# SET ( DESIRED_QT_VERSION "4" CACHE STRING "4: QT4/ 5 not yet implemented" )
# SET ( DONOT_COMPILE_SANDBOX OFF CACHE BOOL "do not compile the SandBox directory" )


# for CGoGN MR
# SET ( FORCE_MR "0" CACHE STRING "0: normal mode / 1 multires mode" )

# #create one big lib
# SET ( ONELIB OFF CACHE BOOL "build CGoGN in one lib" )

# SET ( WITH_GLEWMX OFF CACHE BOOL "use multi-contex GLEW" )

SET ( BUILD_SHARED_LIBS ON )
SET ( WITH_ASSIMP OFF)
SET ( WITH_QT ON)
SET ( DESIRED_QT_VERSION "4")
SET ( DONOT_COMPILE_SANDBOX ON)
SET ( FORCE_MR "0" )
SET ( ONELIB ON)
SET ( WITH_GLEWMX OFF)


IF(WIN32)
	SET ( CMAKE_PREFIX_PATH ${CGoGN_ROOT_DIR}/windows_dependencies CACHE STRING "path to dependencies" )
ENDIF(WIN32)


IF (WITH_QT)
	add_definitions(-DWITH_QT)
	find_package(Qt REQUIRED)
	SET(QT_USE_QTCORE TRUE)
	SET(QT_USE_QTOPENGL TRUE)
	SET(QT_USE_QTSVG TRUE)
	SET(QT_USE_QTXML TRUE )
	INCLUDE(${QT_USE_FILE})
#	ADD_DEFINITIONS(${QT_DEFINITIONS})
#        if(NOT SOFA-LIB_GUI_QGLVIEWER)
#                message(FATAL_ERROR "CgoGN with QT option activated need SOFA-LIB_GUI_QGLVIEWER to be enabled.")
#        endif()
	# find_package(QGLViewer REQUIRED)
        SET (CGoGN_EXT_INCLUDES ${CGoGN_EXT_INCLUDES} ${QT_INCLUDE_DIR} )
        SET (CGoGN_EXT_LIBS ${CGoGN_EXT_LIBS} ${QT_LIBRARIES} )
ENDIF (WITH_QT)


if(NOT (SOFA-EXTERNAL_BOOST_PATH STREQUAL ""))
	message(FATAL_ERROR "CGoGN needs the full boost version. You need to set SOFA-EXTERNAL_BOOST_PATH to \"\"")
endif()
find_package(Boost COMPONENTS regex thread system REQUIRED)

find_package(ZLIB REQUIRED)
find_package(LibXml2 REQUIRED)




IF (DEFINED ASSERTON)
	add_definitions(-DCGOGN_ASSERT_BOOL=${ASSERTON})
ELSE (DEFINED ASSERTON)
	add_definitions(-DCGOGN_ASSERT_BOOL=false)
ENDIF (DEFINED ASSERTON)

IF (FORCE_MR EQUAL 0)
	file(WRITE ${CGoGN_ROOT_DIR}/include/cgogn_mr.h "0" )
ENDIF (FORCE_MR EQUAL 0)

IF (FORCE_MR EQUAL 1)
	add_definitions(-DCGoGN_FORCE_MR=1)
	file(WRITE ${CGoGN_ROOT_DIR}/include/cgogn_mr.h "1" )
ENDIF (FORCE_MR EQUAL 1)

IF (ONELIB)
	file(WRITE ${CGoGN_ROOT_DIR}/include/cgogn_onelib.h "1" )
ELSE (ONELIB)
	file(WRITE ${CGoGN_ROOT_DIR}/include/cgogn_onelib.h "0" )
ENDIF (ONELIB)

add_definitions(-DSHADERPATH="${CGoGN_ROOT_DIR}/lib/Shaders/")
add_definitions("-Werror=return-type")

# define includes of external libs 
SET (CGoGN_EXT_INCLUDES
	${CGoGN_ROOT_DIR}/ThirdParty/include
	${ZLIB_INCLUDE_DIRS}
	${LIBXML2_INCLUDE_DIR}
	${Boost_INCLUDE_DIRS}
	${SOFA-EXTERNAL_EIGEN_PATH}
)

include_directories(
	${CGoGN_ROOT_DIR}/include
	${CMAKE_CURRENT_BINARY_DIR}
	${CGoGN_EXT_INCLUDES}
)


# define libs for external libs
SET (CGoGN_EXT_LIBS
    # nl
	${ZLIB_LIBRARIES}
	${LIBXML2_LIBRARIES}
	${Boost_SYSTEM_LIBRARY}
	${Boost_REGEX_LIBRARY}
	${Boost_THREAD_LIBRARY}
)

#IF(WIN32)
#	SET (CGoGN_EXT_INCLUDES ${CGoGN_EXT_INCLUDES} ${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include)
#	file(GLOB WIN32SUITESPARSELIBS "${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib/*.dll")
#	SET (CGoGN_EXT_LIBS ${CGoGN_EXT_LIBS} 
#						${WIN32SUITESPARSELIBS})
#ELSE(WIN32)
#	find_package(SuiteSparse REQUIRED)
#	SET (CGoGN_EXT_INCLUDES ${CGoGN_EXT_INCLUDES} ${SUITESPARSE_INCLUDE_DIRS})
#	SET (CGoGN_EXT_LIBS ${CGoGN_EXT_LIBS} ${SUITESPARSE_LIBRARIES})
#ENDIF(WIN32)


# IF(APPLE)APPLE
# # attention a changer pour chercher la bonne version automatiquement
# 	SET(CMAKE_OSX_SYSROOT "/Developer/SDKs/MacOSX10.6.sdk" CACHE STRING "developer SDK")
# 	SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined -Wl,dynamic_lookup")
# 	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAC_OSX")
# 	SET(CMAKE_OSX_ARCHITECTURES x86_64)
# 	SET (CGoGN_EXT_LIBS ${CGoGN_EXT_LIBS} "-framework Foundation -framework cocoa" )
# ENDIF(APPLE)

# add_subdirectory(ThirdParty/OpenNL)
# file(GLOB NL_H ThirdParty/OpenNL/src/NL/*.h)
# file(COPY ${NL_H}
# 	 DESTINATION ${CGoGN_ROOT_DIR}/ThirdParty/include/NL)

IF (WITH_ZINRI)
	add_definitions(-DWITH_ZINRI)
	SET (CGoGN_EXT_LIBS ${CGoGN_EXT_LIBS} Zinri)
	add_subdirectory(ThirdParty/Zinri)
	file(GLOB ZINRI_H ThirdParty/Zinri/*.h)
	file(COPY ${ZINRI_H}
		 DESTINATION ${CGoGN_ROOT_DIR}/ThirdParty/include/Zinri)
ENDIF(WITH_ZINRI)


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

file(
	GLOB
	shaders_src
	${CGoGN_ROOT_DIR}/include/Utils/*.frag
	${CGoGN_ROOT_DIR}/include/Utils/*.vert
	${CGoGN_ROOT_DIR}/include/Utils/*.geom	
	${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.frag
	${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.vert
	${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.geom
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

        SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -Wextra -fPIC" )
#        SET( CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS} )
	add_executable( shader_to_h ${CGoGN_ROOT_DIR}/ThirdParty/Tools/shader_to_h.cpp )
	add_custom_target(shader_target shader_to_h ${shaders_src}
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	SOURCES ${shaders_src} )

IF(WIN32)
        add_library(${PROJECT_NAME} ${files_topology} ${files_container} ${files_algo} ${files_utils} ${files_utils_withQt})
	add_dependencies(${PROJECT_NAME} shader_target )
	set(CMAKE_CONFIGURATION_TYPES Release Debug)
	set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Only Release or Debug" FORCE)
#	set(CMAKE_CONFIGURATION_TYPES "Release Debug" CACHE STRING "Only Release or Debug" FORCE)
ELSE(WIN32)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC ")
        message(CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS})
#	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wmissing-field-initializers -Wsign-compare -Wtype-limits -Wuninitialized -fPIC")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Woverride-init -fPIC")
	IF (ONELIB)
        add_library(${PROJECT_NAME} ${files_topology} ${files_container} ${files_algo} ${files_utils} ${files_utils_withQt})
		add_dependencies(${PROJECT_NAME} shader_target )
		SET(CGOGN_LIBS "${PROJECT_NAME}")
	ELSE (ONELIB)
		add_library( ${PROJECT_NAME}			${files_topology})
		add_library( ${PROJECT_NAME}_container	${files_container})
		add_library( ${PROJECT_NAME}_algo		${files_algo})
		add_library( ${PROJECT_NAME}_utils 		${files_utils} )
		SET(CGOGN_LIBS "${PROJECT_NAME}_topology ${PROJECT_NAME}_container ${PROJECT_NAME}_algo ${PROJECT_NAME}_utils")
	ENDIF (ONELIB)
ENDIF(WIN32)

AddLinkerDependencies(${CGoGN_EXT_LIBS} SofaDefaultType ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES}  ${QT_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)
