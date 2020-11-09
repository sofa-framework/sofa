cmake_minimum_required(VERSION 3.12)
project(SofaCommon)

set(SOFACOMMON_TARGETS
    SofaEngine
    SofaExplicitOdeSolver
    SofaImplicitOdeSolver
    SofaLoader
    )

foreach(TARGET ${SOFACOMMON_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
endforeach()

# Modules
set(SOFACOMMON_MODULES
    SofaSimpleFem
    SofaRigid
    SofaDeformable
    SofaObjectInteraction
    SofaMeshCollision
)

foreach(MODULE ${SOFACOMMON_MODULES})
    find_package(${MODULE} REQUIRED)
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
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMMON_TARGETS} ${SOFACOMMON_MODULES})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
