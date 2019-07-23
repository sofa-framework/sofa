cmake_minimum_required(VERSION 3.1)
project(SofaAdvanced)

## Dependencies
find_package(SofaGeneral REQUIRED)
find_package(MiniFlowVR QUIET)
set(SOFA_HAVE_MINIFLOWVR ${MiniFlowVR_FOUND})
set(SOFA_HAVE_FLOWVR ${MiniFlowVR_FOUND}) # cmakedefine

## Version
set(SOFAADVANCED_VERSION ${SofaFramework_VERSION})

## SofaAdvanced/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.h" DESTINATION "include/${PROJECT_NAME}/${PROJECT_NAME}" COMPONENT headers)


### Subdirectories

set(SOFAADVANCED_TARGETS
    SofaNonUniformFem
)

foreach(TARGET ${SOFAADVANCED_TARGETS} SofaComponentAdvanced)
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaAdvanced) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAADVANCED_VERSION})
endforeach()


## CMake package configurations files
sofa_write_package_config_files(SofaAdvanced ${SOFAADVANCED_VERSION})
