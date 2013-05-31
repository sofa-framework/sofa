project("SofaHAPI")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES 
    SofaHAPI.h
    SofaHAPIHapticsDevice.h
    SofaHAPIForceFeedbackEffect.h
    )

set(SOURCE_FILES
    initSofaHAPI.cpp
    SofaHAPIHapticsDevice.cpp
    SofaHAPIForceFeedbackEffect.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
AddCompilerDefinitions("SOFA_BUILD_SOFAHAPI")
AddLinkerDependencies(SofaComponentGeneral SofaHaptics H3DUtil HAPI)
IF( H3D_USE_DEPENDENCIES_ONLY )
  # The variables set here must be set by the CMakeLists.txt that sets H3D_USE_DEPENDENCIES_ONLY to true.
  INCLUDE_DIRECTORIES( ${EXTERNAL_INCLUDE_DIR} ) 
  INCLUDE_DIRECTORIES( ${H3DUTIL_INCLUDE_DIR} ) 
  INCLUDE_DIRECTORIES( ${HAPI_INCLUDE_DIR} ) 
ENDIF( H3D_USE_DEPENDENCIES_ONLY )

include(${SOFA_CMAKE_DIR}/postProject.cmake)