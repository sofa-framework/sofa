cmake_minimum_required(VERSION 3.1)
project(SofaCommon)

## Dependencies
find_package(SofaBase REQUIRED)

## Version
set(SOFACOMMON_VERSION ${SofaFramework_VERSION})

## Subdirectories
set(SOFACOMMON_TARGETS
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
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaCommon) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFACOMMON_VERSION})
endforeach()


set(SOFACOMMON_SRC src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFACOMMON_SRC}/config.h.in
    ${SOFACOMMON_SRC}/initSofaCommon.h
)

set(SOURCE_FILES
    ${SOFACOMMON_SRC}/initSofaCommon.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
foreach(target ${SOFACOMMON_TARGETS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${target})
endforeach()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFACOMMON")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/SofaKernel>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION  ${SOFACOMMON_VERSION})

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${SOFACOMMON_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}/${PROJECT_NAME}"
    )
