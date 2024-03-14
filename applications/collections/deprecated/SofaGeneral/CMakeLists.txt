cmake_minimum_required(VERSION 3.12)
project(SofaGeneral)

# Send warnings when deprecated CMake project is activated
option(CMAKE_WARN_DEPRECATED "Send warning when deprecated projects are found" ON)

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
    SofaGeneralTopology
    SofaTopologyMapping
    SofaUserInteraction
    SofaConstraint
    SofaGeneralLoader
)

foreach(module ${SOFAGENERAL_MODULES})
    sofa_find_package(${module} QUIET)
    if(NOT ${module}_FOUND)
        string(TOUPPER module_${module} MODULE_TO_CHECK)
        list(APPEND LIST_MODULES_TO_CHECK ${MODULE_TO_CHECK})
    endif()
endforeach()
if(LIST_MODULES_TO_CHECK)
    message(FATAL_ERROR
        " If you still want to use ${PROJECT_NAME}, please enable the following CMake option(s): \n"
        " ${LIST_MODULES_TO_CHECK}"
        )
endif()

set(SOFAGENERAL_SRC src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFAGENERAL_SRC}/config.h.in
    ${SOFAGENERAL_SRC}/initSofaGeneral.h
    )

set(SOURCE_FILES
    ${SOFAGENERAL_SRC}/initSofaGeneral.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAGENERAL_MODULES})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
#    RELOCATABLE "collections" # disabled because we need SofaGeneral to be directly available
    )
