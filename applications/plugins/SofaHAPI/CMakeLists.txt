cmake_minimum_required(VERSION 3.22)
project(SofaHAPI)

find_package(SofaUserInteraction REQUIRED)
find_package(SofaHaptics REQUIRED)

set(SOFAHAPI_MAJOR_VERSION 0)
set(SOFAHAPI_MINOR_VERSION 1)
set(SOFAHAPI_VERSION ${SOFAHAPI_MAJOR_VERSION}.${SOFAHAPI_MINOR_VERSION})


set(HEADER_FILES
    SofaHAPIForceFeedbackEffect.h
    SofaHAPIHapticsDevice.h
    config.h.in
)

set(SOURCE_FILES
    SofaHAPIForceFeedbackEffect.cpp
    SofaHAPIHapticsDevice.cpp
    initSofaHAPI.cpp
)

if(NOT TARGET H3DUtil)
    find_package(H3DUtil REQUIRED)
    if(WIN32)
        file(GLOB sharedObjects "${HAPI_DIR}/bin/H3DUtil_*.dll")
        foreach(sharedObject ${sharedObjects})
            file(COPY ${sharedObject} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
        endforeach()
    endif()
else()
    set(H3DUTIL_LIBRARIES H3DUtil)
endif()

if(NOT TARGET HAPI)
    find_package(HAPI REQUIRED)
    if(WIN32)
        # Copy HAPI DLLs
        file(GLOB sharedObjects "${HAPI_DIR}/bin/HAPI_*.dll")

        foreach(sharedObject ${sharedObjects})
            file(COPY ${sharedObject} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
        endforeach()
    endif()
else()
    set(HAPI_LIBRARIES HAPI)
endif()

if(WIN32)
    if(CMAKE_CL_64)
            set(H3D_EXTERNAL_BIN_DIR ${H3D_EXTERNAL_ROOT}/bin64)
        else()
            set(H3D_EXTERNAL_BIN_DIR ${H3D_EXTERNAL_ROOT}/bin32)
    endif()

    file(COPY "${H3D_EXTERNAL_BIN_DIR}/pthreadVC2.dll" DESTINATION
              "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
    file(COPY "${H3D_EXTERNAL_BIN_DIR}/FreeImage.dll" DESTINATION
              "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
endif()

set(SOFAHAPI_DEPENDENCY_INCLUDE_DIR ${HAPI_INCLUDE_DIR}
                                    ${H3DUTIL_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction SofaHaptics ${HAPI_LIBRARIES} ${H3DUTIL_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAHAPI_DEPENDENCY_INCLUDE_DIR}>")

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAHAPI_VERSION})

## Install rules for the library and the header; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME SofaHAPI
    PACKAGE_VERSION ${SOFAHAPI_VERSION}
    TARGETS SofaHAPI AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR SofaHAPI
    )
