# project(CGoGN_ThirdParty)

add_subdirectory(Tools)

# add_subdirectory(PythonQt)
# file(GLOB PYQT_H PythonQt/src/*.h)


# file(COPY ${PYQT_H}
# 	 DESTINATION ${CGoGN_ROOT_DIR}/ThirdParty/include/PythonQt)

add_subdirectory(OpenNL)
file(GLOB NL_H OpenNL/src/NL/*.h)
file(COPY ${NL_H}
	 DESTINATION ${CGoGN_ROOT_DIR}/ThirdParty/include/NL)

IF (WITH_ZINRI)
	add_subdirectory(Zinri)
	file(GLOB ZINRI_H Zinri/*.h)
	file(COPY ${ZINRI_H}
		 DESTINATION ${CGoGN_ROOT_DIR}/ThirdParty/include/Zinri)
ENDIF(WITH_ZINRI)

# IF (WITH_ASSIMP)
# 	add_subdirectory(Assimp)
# 	file(COPY Assimp/include/
# 		 DESTINATION ${CGoGN_ROOT_DIR}/ThirdParty/include/Assimp)
# ENDIF (WITH_ASSIMP)


IF(WIN32)
	SET(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE})
	SET(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "" FORCE)
	MARK_AS_ADVANCED(CMAKE_CONFIGURATION_TYPES)
ENDIF(WIN32)