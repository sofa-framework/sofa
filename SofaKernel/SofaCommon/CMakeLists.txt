cmake_minimum_required(VERSION 3.1)
project(SofaCommon)

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
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
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
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMMON_TARGETS})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFACOMMON")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "${SOFACOMMON_SRC}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
