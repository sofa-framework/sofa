cmake_minimum_required(VERSION 2.8.12)
project(SofaPhysicsAPI)

##
## SofaPhysicsAPI version
##
set( SOFAPHYSICSAPI_MAJOR_VERSION 0 )
set( SOFAPHYSICSAPI_MINOR_VERSION 0 )
set( SOFAPHYSICSAPI_BUILD_VERSION 1 )

set( SOFAPHYSICSAPI_VERSION ${SOFAPHYSICSAPI_MAJOR_VERSION}.${SOFAPHYSICSAPI_MINOR_VERSION}.${SOFAPHYSICSAPI_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(SOFAPHYSICSAPI_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFAPHYSICSAPI_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFAPHYSICSAPI_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAPHYSICSAPI_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaPhysicsAPI)

# Offer the user the choice of overriding the installation directories
set(SOFAPHYSICSAPI_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAPHYSICSAPI_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAPHYSICSAPI_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAPHYSICSAPI_INSTALL_CMAKE_DIR   lib/cmake/SofaPhysicsAPI     CACHE PATH "Installation directory for CMake files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    
    SofaPhysicsAPI.h 
    SofaPhysicsSimulation_impl.h 
    SofaPhysicsOutputMesh_impl.h 
    SofaPhysicsDataMonitor_impl.h 
    SofaPhysicsDataController_impl.h 
    SofaPhysicsOutputMesh_Tetrahedron_impl.h
    fakegui.h
    
    )

set(SOURCE_FILES

    #main.cpp
    SofaPhysicsSimulation.cpp 
    SofaPhysicsOutputMesh.cpp 
    SofaPhysicsDataMonitor.cpp 
    SofaPhysicsDataController.cpp 
    SofaPhysicsOutputMesh_Tetrahedron.cpp
    fakegui.cpp
    
    )

find_package(SofaGui REQUIRED)
find_package(SofaGeneral REQUIRED)
set(SOFAPHYSICSAPI_DEPENDENCY_INCLUDE_DIR ${SofaGui_INCLUDE_DIRS} ${SofaGeneral_INCLUDE_DIRS}) 
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} )
target_link_libraries( ${PROJECT_NAME} PUBLIC SofaGuiMain SofaComponentGeneral ) 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAPHYSICSAPI_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAPHYSICSAPI_COMPILER_FLAGS "-DSOFA_BUILD_SOFAPHYSICSAPI")
# make the name of debug libraries end in _d.
set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAPHYSICSAPI_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAPHYSICSAPI_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

include(CMakePackageConfigHelpers)
# SofaPhysicsAPIConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFAPHYSICSAPI_BUILD_CMAKE_DIR}/SofaPhysicsAPIConfigVersion.cmake
                                 VERSION ${SOFAPHYSICSAPI_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${SOFAPHYSICSAPI_BUILD_CMAKE_DIR}/SofaPhysicsAPIConfigVersion.cmake
        DESTINATION ${SOFAPHYSICSAPI_INSTALL_CMAKE_DIR})


# SofaPhysicsAPIConfig.cmake (build tree)
set(SOFAPHYSICSAPI_INCLUDE_DIR "${SOFAPHYSICSAPI_BUILD_INCLUDE_DIR}")
configure_package_config_file(SofaPhysicsAPIConfig.cmake.in
                              ${SOFAPHYSICSAPI_BUILD_CMAKE_DIR}/SofaPhysicsAPIConfig.cmake
                              INSTALL_DESTINATION ${SOFAPHYSICSAPI_BUILD_CMAKE_DIR}
                              PATH_VARS SOFAPHYSICSAPI_INCLUDE_DIR)

# SofaPhysicsAPIConfig.cmake (installed)
set(SOFAPHYSICSAPI_INCLUDE_DIR "${SOFAPHYSICSAPI_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaPhysicsAPIConfig.cmake.in
                              ${SOFAPHYSICSAPI_BUILD_CMAKE_DIR}/SofaPhysicsAPIConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAPHYSICSAPI_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAPHYSICSAPI_INCLUDE_DIR)


install(TARGETS ${PROJECT_NAME}
    COMPONENT SofaPhysicsAPI_libraries
    EXPORT  SofaPhysicsAPITargets
    RUNTIME DESTINATION "${SOFAPHYSICSAPI_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${SOFAPHYSICSAPI_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${SOFAPHYSICSAPI_INSTALL_LIB_DIR}"
    PUBLIC_HEADER DESTINATION "${SOFAPHYSICSAPI_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
    )

install(EXPORT SofaPhysicsAPITargets DESTINATION ${SOFAPHYSICSAPI_INSTALL_CMAKE_DIR} )
