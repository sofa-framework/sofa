cmake_minimum_required(VERSION 3.1)
project(SofaMisc)

## Dependencies
find_package(SofaGeneral REQUIRED)

## Version
set(SOFAMISC_VERSION ${SofaFramework_VERSION})

## SofaMisc/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/SofaMisc/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/SofaMisc/config.h" DESTINATION "include/SofaMisc")


### Subdirectories

set(SOFAMISC_TARGETS
    SofaComponentMisc
    SofaMisc
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
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaMisc) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAMISC_VERSION})
endforeach()


## CMake package configurations files
sofa_write_package_config_files(SofaMisc ${SOFAMISC_VERSION})
