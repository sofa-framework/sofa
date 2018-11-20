cmake_minimum_required(VERSION 3.1)
project(SofaBase)

## Dependencies
find_package(SofaSimulation REQUIRED)

## Version
set(SOFABASE_VERSION ${SofaFramework_VERSION})

## SofaBase/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/SofaBase/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/SofaBase/config.h" DESTINATION "include/SofaBase" COMPONENT headers)

### Subdirectories

set(SOFABASE_TARGETS
    SofaBaseCollision
    SofaBaseLinearSolver
    SofaBaseMechanics
    SofaBaseTopology
    SofaBaseVisual
    SofaComponentBase
)

foreach(TARGET ${SOFABASE_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaBase) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFABASE_VERSION})
endforeach()

## CMake package configurations files
sofa_write_package_config_files(SofaBase ${SOFABASE_VERSION})
