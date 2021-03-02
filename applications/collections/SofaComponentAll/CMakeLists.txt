cmake_minimum_required(VERSION 3.12)
project(SofaComponentAll VERSION 1.0)

set(SOFACOMPONENTALL_MODULES
    # SofaBase
    SofaBaseLinearSolver
    SofaEigen2Solver
    SofaBaseTopology
    SofaBaseCollision
    SofaBaseMechanics
    SofaBaseVisual
    SofaBaseUtils

    # SofaCommon
    SofaSimpleFem
    SofaRigid
    SofaDeformable
    SofaObjectInteraction
    SofaMeshCollision
    SofaEngine
    SofaExplicitOdeSolver
    SofaImplicitOdeSolver
    SofaLoader

    # SofaGeneral
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

    # SofaAdvanced
    SofaNonUniformFem

    # SofaMisc
    SofaMiscExtra
    SofaMiscEngine
    SofaMiscFem
    SofaMiscForceField
    SofaMiscMapping
    SofaMiscSolver
    SofaMiscTopology
    )

set(SOFACOMPONENTALL_MODULES_INCLUDES "")
set(SOFACOMPONENTALL_MODULES_INITS "")
foreach(module ${SOFACOMPONENTALL_MODULES})
    sofa_find_package(${module} REQUIRED)
    set(SOFACOMPONENTALL_MODULES_INCLUDES "${SOFACOMPONENTALL_MODULES_INCLUDES}\n#include <${module}/init${module}.h>")
    set(SOFACOMPONENTALL_MODULES_INITS "${SOFACOMPONENTALL_MODULES_INITS} \\\n    init${module}();")
endforeach()

set(HEADER_FILES
    config.h.in
    components.h.in
    initSofaComponentAll.h
    )
set(SOURCE_FILES
    initSofaComponentAll.cpp
    )
set(EXTRA_FILES
    README.md
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} ${SOFACOMPONENTALL_MODULES})

find_package(SofaOpenglVisual QUIET)
if(SofaOpenglVisual_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaOpenglVisual)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "SofaComponentAll"
    )
