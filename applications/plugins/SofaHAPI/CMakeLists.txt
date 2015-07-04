project("SofaHAPI")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES 
    initSofaHAPI.h
    SofaHAPIHapticsDevice.h
    SofaHAPIForceFeedbackEffect.h
    )

set(SOURCE_FILES
    initSofaHAPI.cpp
    SofaHAPIHapticsDevice.cpp
    SofaHAPIForceFeedbackEffect.cpp
    )


find_package(HAPI REQUIRED)
include_directories(${HAPI_INCLUDE_DIR})
find_package(H3DUtil REQUIRED)
include_directories(${H3DUTIL_INCLUDE_DIR})

if(WIN32)

    #copy required HAPI dlls
    file(GLOB sharedObjects "${HAPI_DIR}/bin/HAPI_*.dll")

    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()

    file(GLOB sharedObjects "${HAPI_DIR}/bin/H3DUtil_*.dll")

    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()

    file(COPY "${HAPI_DIR}/External/bin/pthreadVC2.dll" DESTINATION "${SOFA_BIN_DIR}")
    file(COPY "${HAPI_DIR}/External/bin/FreeImage.dll" DESTINATION "${SOFA_BIN_DIR}")

    #file(COPY "${HAPI_DIR}/bin/HAPI_*.dll" DESTINATION ${SOFA_BIN_DIR})
    #file(COPY "${HAPI_DIR}/bin/H3DUtil_*.dll" DESTINATION ${SOFA_BIN_DIR})
    #file(COPY "${HAPI_DIR}/External/bin/pthreadVC2.dll" DESTINATION ${SOFA_BIN_DIR})
    #file(COPY "${HAPI_DIR}/External/bin/FreeImage.dll" DESTINATION ${SOFA_BIN_DIR})

endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
AddCompilerDefinitions("SOFA_BUILD_SOFAHAPI")
AddLinkerDependencies(SofaComponentGeneral SofaHaptics ${HAPI_LIBRARIES} ${H3DUTIL_LIBRARIES} )


IF( H3D_USE_DEPENDENCIES_ONLY )
  # The variables set here must be set by the CMakeLists.txt that sets H3D_USE_DEPENDENCIES_ONLY to true.
  INCLUDE_DIRECTORIES( ${EXTERNAL_INCLUDE_DIR} ) 
  INCLUDE_DIRECTORIES( ${H3DUTIL_INCLUDE_DIR} ) 
  INCLUDE_DIRECTORIES( ${HAPI_INCLUDE_DIR} ) 
ENDIF( H3D_USE_DEPENDENCIES_ONLY )

include(${SOFA_CMAKE_DIR}/postProject.cmake)