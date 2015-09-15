cmake_minimum_required(VERSION 2.8.12)
project(SofaAdvanced)

## Dependencies
find_package(SofaGeneral REQUIRED)
find_package(MiniFlowVR)
set(SOFA_HAVE_MINIFLOWVR ${MiniFlowVR_FOUND})
set(SOFA_HAVE_FLOWVR ${MiniFlowVR_FOUND}) # cmakedefine

## Version
set(SOFAADVANCED_VERSION ${SofaFramework_VERSION})

## SofaAdvanced/config.h
configure_file(config.h.in SofaAdvanced/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SofaAdvanced/config.h DESTINATION "include/SofaAdvanced")

## Installation macro for subdirectories
macro(SofaAdvanced_install lib)
    install(TARGETS ${lib}
            EXPORT SofaAdvancedTargets
            RUNTIME DESTINATION bin COMPONENT SofaAdvanced_libraries
            LIBRARY DESTINATION lib COMPONENT SofaAdvanced_libraries
            ARCHIVE DESTINATION lib COMPONENT SofaAdvanced_libraries
            PUBLIC_HEADER DESTINATION include/${lib} COMPONENT SofaAdvanced_headers)
endmacro()


### Subdirectories

set(SOFAADVANCED_TARGETS
    SofaComponentAdvanced
    SofaEulerianFluid
    SofaNonUniformFem
    SofaSphFluid
    SofaVolumetricData
)

foreach(TARGET ${SOFAADVANCED_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>") # For SofaAdvanced/config.h
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${SofaGeneral_INCLUDE_DIRS}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaAdvanced) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAADVANCED_VERSION})
endforeach()


## CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaAdvanced ${SOFAADVANCED_VERSION} "" "include")
