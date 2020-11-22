cmake_minimum_required(VERSION 3.12)
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
    )

foreach(TARGET ${SOFACOMMON_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
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

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
