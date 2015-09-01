cmake_minimum_required(VERSION 2.8.12)
project(SofaPhysicsAPI)

##
## SofaPhysicsAPI version
##
set(SOFAPHYSICSAPI_MAJOR_VERSION 0)
set(SOFAPHYSICSAPI_MINOR_VERSION 0)
set(SOFAPHYSICSAPI_BUILD_VERSION 1)

set(SOFAPHYSICSAPI_VERSION ${SOFAPHYSICSAPI_MAJOR_VERSION}.${SOFAPHYSICSAPI_MINOR_VERSION}.${SOFAPHYSICSAPI_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")


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

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiMain SofaComponentGeneral)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAPHYSICSAPI_DEPENDENCY_INCLUDE_DIR}>")

set(SOFAPHYSICSAPI_COMPILER_FLAGS "-DSOFA_BUILD_SOFAPHYSICSAPI")
# make the name of debug libraries end in _d.
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAPHYSICSAPI_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAPHYSICSAPI_COMPILER_FLAGS}")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

include(CMakePackageConfigHelpers)
# SofaPhysicsAPIConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(SofaPhysicsAPIConfigVersion.cmake
                                 VERSION ${SOFAPHYSICSAPI_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaPhysicsAPIConfigVersion.cmake
        DESTINATION lib/cmake/SofaPhysicsAPI)


# SofaPhysicsAPIConfig.cmake (build tree)
set(SOFAPHYSICSAPI_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/SofaPhysicsAPIConfig.cmake.in
                              SofaPhysicsAPIConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SOFAPHYSICSAPI_INCLUDE_DIR)

# SofaPhysicsAPIConfig.cmake (installed)
set(SOFAPHYSICSAPI_INCLUDE_DIR include)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/SofaPhysicsAPIConfig.cmake.in
                              SofaPhysicsAPIConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/SofaPhysicsAPI
                              PATH_VARS SOFAPHYSICSAPI_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaPhysicsAPIConfigInstall.cmake
        DESTINATION lib/cmake/SofaPhysicsAPI
        RENAME SofaPhysicsAPIConfig.cmake)

install(TARGETS ${PROJECT_NAME}
    COMPONENT SofaPhysicsAPI_libraries
    EXPORT SofaPhysicsAPITargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")

install(EXPORT SofaPhysicsAPITargets DESTINATION lib/cmake/SofaPhysicsAPI)
