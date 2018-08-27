cmake_minimum_required(VERSION 3.1)
project(SofaCommon)

## Dependencies
find_package(SofaBase REQUIRED)

## Version
set(SOFACOMMON_VERSION ${SofaFramework_VERSION})

## SofaCommon/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/SofaCommon/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/SofaCommon/config.h" DESTINATION "include/SofaCommon" COMPONENT headers)


### Subdirectories

set(SOFACOMMON_TARGETS
    SofaComponentCommon
    SofaDeformable
    SofaEngine
    SofaExplicitOdeSolver
    SofaImplicitOdeSolver
    SofaLoader
    SofaMeshCollision
    SofaObjectInteraction
    SofaRigid
    SofaSimpleFem
    SofaEigen2Solver
)

foreach(TARGET ${SOFACOMMON_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaCommon) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFACOMMON_VERSION})
endforeach()


## CMake package configurations files
sofa_write_package_config_files(SofaCommon ${SOFACOMMON_VERSION})
