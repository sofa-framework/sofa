cmake_minimum_required(VERSION 3.1)
project(SofaGeneral)

# Send warnings when deprecated CMake project is activated
option(CMAKE_WARN_DEPRECATED "Send warning when deprecated projects are found" ON)

# Subdirectories
set(SOFAGENERAL_TARGETS
    SofaGeneralAnimationLoop
    SofaGeneralDeformable
    SofaGeneralExplicitOdeSolver
    SofaGeneralImplicitOdeSolver
    SofaGeneralLinearSolver
    SofaGeneralLoader
    SofaGeneralMeshCollision
    SofaGeneralObjectInteraction
    SofaGeneralRigid
    SofaGeneralSimpleFem
    SofaGeneralTopology
    SofaGeneralVisual
    SofaBoundaryCondition
    SofaConstraint
    SofaGeneralEngine
    SofaGraphComponent
    SofaTopologyMapping
    SofaUserInteraction
)

foreach(TARGET ${SOFAGENERAL_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../modules>")
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
endforeach()

set(SOFAGENERAL_SRC src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFAGENERAL_SRC}/config.h.in
    ${SOFAGENERAL_SRC}/initSofaGeneral.h
    )

set(SOURCE_FILES
    ${SOFAGENERAL_SRC}/initSofaGeneral.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAGENERAL_TARGETS})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGENERAL")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "${SOFAGENERAL_SRC}"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
