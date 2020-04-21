cmake_minimum_required(VERSION 3.1)
project(SofaGeneral)

# Send warnings when deprecated CMake project is activated
option(CMAKE_WARN_DEPRECATED "Send warning when deprecated projects are found" ON)

# Dependencies
find_package(SofaCommon REQUIRED)

# Version
set(SOFAGENERAL_VERSION ${SofaFramework_VERSION})

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
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
    set_target_properties(${TARGET} PROPERTIES DEBUG_POSTFIX _d)
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaGeneral) # IDE folder
    set_target_properties(${TARGET} PROPERTIES VERSION  ${SOFAGENERAL_VERSION})
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

foreach(target ${SOFAGENERAL_TARGETS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${target})
endforeach()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGENERAL")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/SofaKernel>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION  ${SOFAGENERAL_VERSION})

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${SOFAGENERAL_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}/${PROJECT_NAME}"
)

# SofaGeneral/config.h
# configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.h")
# install(FILES "${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/config.h" DESTINATION "include/${PROJECT_NAME}/${PROJECT_NAME}" COMPONENT headers)
