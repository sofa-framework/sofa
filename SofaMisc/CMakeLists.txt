cmake_minimum_required(VERSION 2.8.12)
project(SofaMisc)

## Dependencies
find_package(SofaGeneral REQUIRED)

## Version
set(SOFAMISC_VERSION ${SofaFramework_VERSION})

## SofaMisc/config.h
configure_file(config.h.in SofaMisc/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaMisc/config.h DESTINATION "include/SofaMisc")

## Installation macro for subdirectories
macro(SofaMisc_install lib)
    install(TARGETS ${lib}
            EXPORT SofaMiscTargets
            RUNTIME DESTINATION bin COMPONENT SofaMisc_libraries
            LIBRARY DESTINATION lib COMPONENT SofaMisc_libraries
            ARCHIVE DESTINATION lib COMPONENT SofaMisc_libraries
            PUBLIC_HEADER DESTINATION include/${lib} COMPONENT SofaMisc_headers)
endmacro()


### Subdirectories

set(SOFAMISC_TARGETS
    SofaComponentMisc
    SofaMisc
    SofaMiscCollision
    SofaMiscEngine
    SofaMiscFem
    SofaMiscForceField
    SofaMiscMapping
    SofaMiscSolver
    SofaMiscTopology
)

foreach(TARGET ${SOFAMISC_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>") # For SofaMisc/config.h
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${SofaGeneral_INCLUDE_DIRS}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaMisc) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAMISC_VERSION})
endforeach()


## CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaMisc ${SOFAMISC_VERSION} "" "include")
