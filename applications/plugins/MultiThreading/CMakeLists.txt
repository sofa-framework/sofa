cmake_minimum_required(VERSION 2.8.12)
project(MultiThreading)

##
## MultiThreading version
##
set( MULTITHREADING_MAJOR_VERSION 0 )
set( MULTITHREADING_MINOR_VERSION 0 )
set( MULTITHREADING_BUILD_VERSION 1 )

set( MULTITHREADING_VERSION ${MULTITHREADING_MAJOR_VERSION}.${MULTITHREADING_MINOR_VERSION}.${MULTITHREADING_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")
set(MULTITHREADING_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(MULTITHREADING_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(MULTITHREADING_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MULTITHREADING_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/MultiThreading)

# Offer the user the choice of overriding the installation directories
set(MULTITHREADING_INSTALL_LIB_DIR     lib                        CACHE PATH "Installation directory for libraries")
set(MULTITHREADING_INSTALL_BIN_DIR     bin                        CACHE PATH "Installation directory for executables")
set(MULTITHREADING_INSTALL_INCLUDE_DIR include                    CACHE PATH "Installation directory for header files")
set(MULTITHREADING_INSTALL_CMAKE_DIR   ${MULTITHREADING_INSTALL_LIB_DIR}/cmake/MultiThreading )

set(HEADER_FILES
    src/TaskSchedulerBoost.h
    src/Tasks.h
    src/Tasks.inl
    src/initMultiThreading.h
    )

set(SOURCE_FILES
    src/TaskSchedulerBoost.cpp
    src/Tasks.cpp
    src/initMultiThreading.cpp
    )

find_package(SofaGeneral REQUIRED)

set(MULTITHREADING_DEPENDENCY_INCLUDE_DIR ${SofaGeneral_INCLUDE_DIRS})
set(MULTITHREADING_DEPENDENCIES_LINK SofaHelper)

set(MULTITHREADING_COMPILER_FLAGS "-DSOFA_MULTITHREADING_PLUGIN")



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} PUBLIC ${MULTITHREADING_DEPENDENCIES_LINK} )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${MULTITHREADING_DEPENDENCY_INCLUDE_DIR}>" )

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${MULTITHREADING_COMPILER_FLAGS}"  )
if(NOT SOFA_INSTALL_HEADERS_GENERATED_ONLY)
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )
endif()

# if(TARGET gtest)
#     add_subdirectory(MultiThreading_test)
# endif()
include(CMakePackageConfigHelpers)
# MultiThreadingConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${MULTITHREADING_BUILD_CMAKE_DIR}/MultiThreadingConfigVersion.cmake
                                 VERSION ${MULTITHREADING_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${MULTITHREADING_BUILD_CMAKE_DIR}/MultiThreadingConfigVersion.cmake
        DESTINATION ${MULTITHREADING_INSTALL_CMAKE_DIR})
# MultiThreadingConfig.cmake (build tree)
set(MULTITHREADING_INCLUDE_DIR "${MULTITHREADING_BUILD_INCLUDE_DIR}")
configure_package_config_file(MultiThreadingConfig.cmake.in
                              ${MULTITHREADING_BUILD_CMAKE_DIR}/MultiThreadingConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS MULTITHREADING_INCLUDE_DIR)
# MultiThreadingConfig.cmake (installed)
set(MULTITHREADING_INCLUDE_DIR "${MULTITHREADING_INSTALL_INCLUDE_DIR}")
configure_package_config_file(MultiThreadingConfig.cmake.in
                              ${MULTITHREADING_BUILD_CMAKE_DIR}/MultiThreadingConfigInstall.cmake
                              INSTALL_DESTINATION ${MULTITHREADING_INSTALL_CMAKE_DIR}
                              PATH_VARS MULTITHREADING_INCLUDE_DIR )
install(FILES ${MULTITHREADING_BUILD_CMAKE_DIR}/MultiThreadingConfigInstall.cmake
        DESTINATION ${MULTITHREADING_INSTALL_CMAKE_DIR}
        RENAME MultiThreadingConfig.cmake)

install(TARGETS ${PROJECT_NAME}
        EXPORT    MultiThreadingTargets
        RUNTIME DESTINATION "${MULTITHREADING_INSTALL_BIN_DIR}" COMPONENT MultiThreading_libraries
        LIBRARY DESTINATION "${MULTITHREADING_INSTALL_LIB_DIR}" COMPONENT MultiThreading_libraries
        ARCHIVE DESTINATION "${MULTITHREADING_INSTALL_LIB_DIR}" COMPONENT MultiThreading_libraries
        PUBLIC_HEADER DESTINATION "${MULTITHREADING_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT MultiThreading_headers)

# MultiThreadingTargets.cmake (installed)
install(EXPORT MultiThreadingTargets
        DESTINATION ${MULTITHREADING_INSTALL_CMAKE_DIR} )
