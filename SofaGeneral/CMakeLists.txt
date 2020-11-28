cmake_minimum_required(VERSION 3.12)
project(SofaGeneral)

# Send warnings when deprecated CMake project is activated
option(CMAKE_WARN_DEPRECATED "Send warning when deprecated projects are found" ON)

# Subdirectories
set(SOFAGENERAL_TARGETS
)

foreach(TARGET ${SOFAGENERAL_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
endforeach()

# Modules
set(SOFAGENERAL_MODULES
    SofaBoundaryCondition
    SofaGeneralMeshCollision
    SofaGeneralVisual
    SofaGraphComponent
    SofaGeneralAnimationLoop
    SofaGeneralDeformable
    SofaGeneralEngine
    SofaGeneralExplicitOdeSolver
    SofaGeneralImplicitOdeSolver
    SofaGeneralLinearSolver
    SofaGeneralRigid
    SofaGeneralObjectInteraction
    SofaGeneralSimpleFem
    SofaGeneralTopology
    SofaTopologyMapping
    SofaUserInteraction
    SofaConstraint
    SofaGeneralLoader
)
foreach(MODULE ${SOFAGENERAL_MODULES})
    find_package(${MODULE} REQUIRED)
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
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAGENERAL_TARGETS} ${SOFAGENERAL_MODULES})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
