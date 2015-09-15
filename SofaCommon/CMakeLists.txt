cmake_minimum_required(VERSION 2.8.12)
project(SofaCommon)

## Dependencies
find_package(SofaBase REQUIRED)

## Version
set(SOFACOMMON_VERSION ${SofaFramework_VERSION})

## SofaCommon/config.h
configure_file(config.h.in SofaCommon/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaCommon/config.h DESTINATION "include/SofaCommon")

## Installation macro for subdirectories
macro(SofaCommon_install lib)
    install(TARGETS ${lib}
            EXPORT SofaCommonTargets
            RUNTIME DESTINATION bin COMPONENT SofaCommon_libraries
            LIBRARY DESTINATION lib COMPONENT SofaCommon_libraries
            ARCHIVE DESTINATION lib COMPONENT SofaCommon_libraries
            PUBLIC_HEADER DESTINATION include/${lib} COMPONENT SofaCommon_headers)
endmacro()


### Subdirectories

set(SOFACOMMON_TARGETS
    SofaComponentCommon
    SofaDeformable
    SofaExplicitOdeSolver
    SofaImplicitOdeSolver
    SofaLoader
    SofaMeshCollision
    SofaObjectInteraction
    SofaRigid
    SofaSimpleFem
)

foreach(TARGET ${SOFACOMMON_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>") # For SofaCommon/config.h
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${SofaBase_INCLUDE_DIRS}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaCommon) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFACOMMON_VERSION})
endforeach()


## CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaCommon ${SOFACOMMON_VERSION} "" "include")
