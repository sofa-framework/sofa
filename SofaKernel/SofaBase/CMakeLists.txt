cmake_minimum_required(VERSION 3.1)
project(SofaBase)

set(SOFABASE_TARGETS
    SofaBaseCollision
    SofaBaseLinearSolver
    SofaBaseMechanics
    SofaBaseTopology
    SofaBaseVisual
    SofaBaseUtils
    )

foreach(TARGET ${SOFABASE_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
endforeach()

set(SOFABASE_SRC src/${PROJECT_NAME})
set(HEADER_FILES
    ${SOFABASE_SRC}/config.h.in
    ${SOFABASE_SRC}/initSofaBase.h
    )
set(SOURCE_FILES
    ${SOFABASE_SRC}/initSofaBase.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFABASE_TARGETS})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFABASE")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} 
    INCLUDE_SOURCE_DIR "${SOFABASE_SRC}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
