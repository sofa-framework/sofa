include(${SOFA_CMAKE_DIR}/preProject.cmake)


SET(CGoGN_ROOT_DIR ${SOFA_EXTLIBS_DIR}/CGoGN )
LIST (APPEND CMAKE_MODULE_PATH ${CGoGN_ROOT_DIR}/cmake_modules/ )

#SET ( CMAKE_VERBOSE_MAKEFILE 1 )
SET ( CGOGN_WITH_QT ON)
SET ( CGOGN_DESIRED_QT_VERSION "4")
SET ( CGoGN_COMPILE_SANDBOX OFF)
SET ( CGoGN_ONELIB ON)
SET ( CGOGN_WITH_GLEWMX OFF)
SET ( CGOGN_WITH_ZINRI OFF)
SET ( CGoGN_COMPILE_BENCHES OFF)
SET ( CGoGN_ASSERT_ACTIVED OFF)

IF(WIN32)
	SET ( CMAKE_PREFIX_PATH ${CGoGN_ROOT_DIR}/windows_dependencies CACHE STRING "path to dependencies" )
ENDIF(WIN32)

SET (CGoGN_EXT_INCLUDES "")
SET (CGoGN_EXT_LIBS "")

find_package(Boost COMPONENTS regex thread system REQUIRED)
find_package(ZLIB REQUIRED)
find_package(LibXml2 REQUIRED)

IF (CGOGN_WITH_QT)
        add_definitions("-DCGOGN_WITH_QT")
	SET(QT_USE_QTCORE TRUE)
	SET(QT_USE_QTOPENGL TRUE)
	SET(QT_USE_QTSVG TRUE)
	SET(QT_USE_QTXML TRUE )
        find_package(Qt REQUIRED)
	INCLUDE(${QT_USE_FILE})
        LIST (APPEND CGoGN_EXT_INCLUDES ${QT_INCLUDE_DIR} )
        LIST (APPEND CGoGN_EXT_LIBS ${QT_LIBRARIES} )
ENDIF (CGOGN_WITH_QT)

if(NOT (SOFA-EXTERNAL_BOOST_PATH STREQUAL ""))
	message(FATAL_ERROR "CGoGN needs the full boost version. You need to set SOFA-EXTERNAL_BOOST_PATH to \"\"")
endif()

add_definitions("-DSHADERPATH=\"${CGoGN_ROOT_DIR}/lib/Shaders/\"")
add_definitions("-Werror=return-type")
add_definitions("-DGLM_FORCE_RADIANS")
add_definitions("-Wno-unused-local-typedef")
IF (APPLE)
        add_definitions("-DCGOGN_APPLE")
ENDIF ()

IF (DEFINED CGoGN_ASSERT_ACTIVED)
        add_definitions("-DCGOGN_ASSERT_BOOL=${CGoGN_ASSERT_ACTIVED}")
ELSE (DEFINED CGoGN_ASSERT_ACTIVED)
        add_definitions("-DCGOGN_ASSERT_BOOL=false")
ENDIF (DEFINED CGoGN_ASSERT_ACTIVED)

# define includes of external libs 
LIST (APPEND CGoGN_EXT_INCLUDES
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
LIST (APPEND CGoGN_EXT_LIBS
	${ZLIB_LIBRARIES}
	${LIBXML2_LIBRARIES}
	${Boost_SYSTEM_LIBRARY}
	${Boost_REGEX_LIBRARY}
	${Boost_THREAD_LIBRARY}
)

IF (CGOGN_WITH_ZINRI)
        add_definitions("-DCGOGN_WITH_ZINRI")
        LIST (APPEND CGoGN_EXT_LIBS  Zinri)
	add_subdirectory(ThirdParty/Zinri)
ENDIF(CGOGN_WITH_ZINRI)


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

file(	GLOB
	shaders_src
        ${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.frag
        ${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.vert
        ${CGoGN_ROOT_DIR}/include/Utils/Shaders/*.geom
        ${CGoGN_ROOT_DIR}/include/Utils/*.frag
        ${CGoGN_ROOT_DIR}/include/Utils/*.vert
        ${CGoGN_ROOT_DIR}/include/Utils/*.geom
)

add_executable( shader_to_h ${CGoGN_ROOT_DIR}/ThirdParty/Tools/shader_to_h.cpp )

add_custom_target(shader_target shader_to_h ${shaders_src}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  SOURCES ${shaders_src}
                 )

add_dependencies(shader_target shader_to_h) #ensure that shader_to_h is compiled

IF(CGOGN_WITH_QT)
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
ENDIF (CGOGN_WITH_QT)

IF(NOT WIN32)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC" )
    IF(CMAKE_COMPILER_IS_GNUCXX)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-local-typedefs")
    ENDIF()
    IF(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wno-overloaded-virtual")
    ENDIF()
ELSE()
    ADD_DEFINITIONS("/wd4251") #dll export warnings
ENDIF()


IF(WIN32)
        add_library(${PROJECT_NAME} ${files_topology} ${files_container} ${files_algo} ${files_utils} ${files_utils_withQt})
	add_dependencies(${PROJECT_NAME} shader_target )
	set(CMAKE_CONFIGURATION_TYPES Release Debug)
	set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Only Release or Debug" FORCE)
#	set(CMAKE_CONFIGURATION_TYPES "Release Debug" CACHE STRING "Only Release or Debug" FORCE)
ELSE(WIN32)
        IF (CGoGN_ONELIB)
            add_library(${PROJECT_NAME} ${files_topology} ${files_container} ${files_algo} ${files_utils} ${files_utils_withQt})
            add_dependencies(${PROJECT_NAME} shader_target )
            SET(CGOGN_LIBS "${PROJECT_NAME}")
        ELSE (CGoGN_ONELIB)
            add_library( ${PROJECT_NAME}			${files_topology})
            add_library( ${PROJECT_NAME}_container	${files_container})
            add_library( ${PROJECT_NAME}_algo		${files_algo})
            add_library( ${PROJECT_NAME}_utils 		${files_utils} )
            SET(CGOGN_LIBS "${PROJECT_NAME}_topology ${PROJECT_NAME}_container ${PROJECT_NAME}_algo ${PROJECT_NAME}_utils")
            add_dependencies(${PROJECT_NAME}_utils shader_target )
        ENDIF (CGoGN_ONELIB)
ENDIF(WIN32)

AddLinkerDependencies(${CGoGN_EXT_LIBS} SofaDefaultType ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES}  ${QT_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)
