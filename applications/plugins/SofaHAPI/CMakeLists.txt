cmake_minimum_required(VERSION 2.8.12)
project(SofaHAPI)

##
## SofaHAPI version
##
set( SOFAHAPI_MAJOR_VERSION 0 )
set( SOFAHAPI_MINOR_VERSION 0 )
set( SOFAHAPI_BUILD_VERSION 1 )

set( SOFAHAPI_VERSION 
    ${SOFAHAPI_MAJOR_VERSION}.${SOFAHAPI_MINOR_VERSION}.${SOFAHAPI_BUILD_VERSION} 
    CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(SOFAHAPI_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFAHAPI_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFAHAPI_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAHAPI_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaHAPI)

# Offer the user the choice of overriding the installation directories
set(SOFAHAPI_INSTALL_LIB_DIR     lib                    CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAHAPI_INSTALL_BIN_DIR     bin                    CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAHAPI_INSTALL_INCLUDE_DIR include                CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAHAPI_INSTALL_CMAKE_DIR   lib/cmake/SofaHAPI     CACHE PATH "Installation directory for CMake files relative to CMAKE_INSTALL_PREFIX variable")

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
         #copy required HAPI dlls
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

set(SOFAHAPI_DEPENDENCY_INCLUDE_DIR ${SofaGeneral_INCLUDE_DIRS} 
                                    ${HAPI_INCLUDE_DIR} 
                                    ${H3DUTIL_INCLUDE_DIR} )

if( H3D_USE_DEPENDENCIES_ONLY )
  # The variables set here must be set by the CMakeLists.txt that sets H3D_USE_DEPENDENCIES_ONLY to true.
  list(APPEND SOFAHAPI_DEPENDENCY_INCLUDE_DIR ${EXTERNAL_INCLUDE_DIR} )
endif()




add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} PUBLIC SofaUserInteraction SofaHaptics 
                       ${HAPI_LIBRARIES} ${H3DUTIL_LIBRARIES} )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAHAPI_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAHAPI_COMPILER_FLAGS "-DSOFA_BUILD_SOFAHAPI")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAHAPI_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAHAPI_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SofaHAPI.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaHAPI.h)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaHAPI.h
         DESTINATION "${SOFAHAPI_INSTALL_INCLUDE_DIR}/sofa")

include(CMakePackageConfigHelpers)
# SofaHAPIConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFAHAPI_BUILD_CMAKE_DIR}/SofaHAPIConfigVersion.cmake
                                 VERSION ${SOFAHAPI_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${SOFAHAPI_BUILD_CMAKE_DIR}/SofaHAPIConfigVersion.cmake
        DESTINATION ${SOFAHAPI_INSTALL_CMAKE_DIR})


# SofaHAPIConfig.cmake (build tree)
set(SOFAHAPI_INCLUDE_DIR "${SOFAHAPI_BUILD_INCLUDE_DIR}")
configure_package_config_file(SofaHAPIConfig.cmake.in
                              ${SOFAHAPI_BUILD_CMAKE_DIR}/SofaHAPIConfig.cmake
                              INSTALL_DESTINATION ${SOFAHAPI_BUILD_CMAKE_DIR}
                              PATH_VARS SOFAHAPI_INCLUDE_DIR)

# SofaHAPIConfig.cmake (installed)
set(SOFAHAPI_INCLUDE_DIR "${SOFAHAPI_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaHAPIConfig.cmake.in
                              ${SOFAHAPI_BUILD_CMAKE_DIR}/SofaHAPIConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAHAPI_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAHAPI_INCLUDE_DIR)
                              
install(TARGETS ${PROJECT_NAME}
    COMPONENT SofaHAPI_libraries
    EXPORT  SofaHAPITargets
    RUNTIME DESTINATION "${SOFAHAPI_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${SOFAHAPI_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${SOFAHAPI_INSTALL_LIB_DIR}"
    PUBLIC_HEADER DESTINATION "${SOFAHAPI_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
    )

