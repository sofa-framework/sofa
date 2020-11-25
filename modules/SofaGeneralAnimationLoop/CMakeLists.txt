cmake_minimum_required(VERSION 3.12)
project(SofaGeneralAnimationLoop LANGUAGES CXX)

set(SOFAGENERALANIMATIONLOOP_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALANIMATIONLOOP_SRC}/config.h.in
    ${SOFAGENERALANIMATIONLOOP_SRC}/initSofaGeneralAnimationLoop.h
    )

set(SOURCE_FILES
    ${SOFAGENERALANIMATIONLOOP_SRC}/initSofaGeneralAnimationLoop.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALANIMATIONLOOP_SRC}/MultiStepAnimationLoop.h
    ${SOFAGENERALANIMATIONLOOP_SRC}/MultiTagAnimationLoop.h
    ${SOFAGENERALANIMATIONLOOP_SRC}/MechanicalMatrixMapper.h
    ${SOFAGENERALANIMATIONLOOP_SRC}/MechanicalMatrixMapper.inl)

list(APPEND SOURCE_FILES
    ${SOFAGENERALANIMATIONLOOP_SRC}/MultiStepAnimationLoop.cpp
    ${SOFAGENERALANIMATIONLOOP_SRC}/MultiTagAnimationLoop.cpp
    ${SOFAGENERALANIMATIONLOOP_SRC}/MechanicalMatrixMapper.cpp)

find_package(SofaBase REQUIRED) # SofaBaseLinearSolver

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon SofaBaseLinearSolver)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
