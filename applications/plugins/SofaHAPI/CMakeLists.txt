cmake_minimum_required(VERSION 2.8.12)
project(SofaHAPI)

set(SOFAHAPI_MAJOR_VERSION 0)
set(SOFAHAPI_MINOR_VERSION 1)
set(SOFAHAPI_VERSION ${SOFAHAPI_MAJOR_VERSION}.${SOFAHAPI_MINOR_VERSION})


set(HEADER_FILES
    SofaHAPIForceFeedbackEffect.h
    SofaHAPIHapticsDevice.h
)

set(SOURCE_FILES
    SofaHAPIForceFeedbackEffect.cpp
    SofaHAPIHapticsDevice.cpp
    initSofaHAPI.cpp
)

find_package(SofaGeneral REQUIRED)

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

if(H3D_USE_DEPENDENCIES_ONLY)
    # The variables set here must be set by the CMakeLists.txt that sets H3D_USE_DEPENDENCIES_ONLY to true.
    list(APPEND SOFAHAPI_DEPENDENCY_INCLUDE_DIR ${EXTERNAL_INCLUDE_DIR})
endif()

    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()

    file(COPY "${HAPI_DIR}/External/bin/pthreadVC2.dll" DESTINATION "${SOFA_BIN_DIR}")
    file(COPY "${HAPI_DIR}/External/bin/FreeImage.dll" DESTINATION "${SOFA_BIN_DIR}")

    # file(COPY "${HAPI_DIR}/bin/HAPI_*.dll" DESTINATION ${SOFA_BIN_DIR})
    # file(COPY "${HAPI_DIR}/bin/H3DUtil_*.dll" DESTINATION ${SOFA_BIN_DIR})
    # file(COPY "${HAPI_DIR}/External/bin/pthreadVC2.dll" DESTINATION ${SOFA_BIN_DIR})
    # file(COPY "${HAPI_DIR}/External/bin/FreeImage.dll" DESTINATION ${SOFA_BIN_DIR})

endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction SofaHaptics ${HAPI_LIBRARIES} ${H3DUTIL_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAHAPI_DEPENDENCY_INCLUDE_DIR}>")

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAHAPI_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAHAPI")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

## SofaHAPI/config.h
configure_file(SofaHAPI.h.in "${CMAKE_BINARY_DIR}/include/SofaHAPI/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/SofaHAPI/config.h" DESTINATION "include/SofaHAPI")

## Install rules for the library and the header; CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_create_package(SofaHAPI ${SOFAHAPI_VERSION} SofaHAPI SofaHAPI)

## Install rules for the resources
install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

