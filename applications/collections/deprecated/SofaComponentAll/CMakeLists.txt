cmake_minimum_required(VERSION 3.12)
project(SofaComponentAll VERSION 1.0)

set(SOFACOMPONENTALL_MODULES
    # SofaBase
    SofaBaseCollision
    SofaBaseLinearSolver
    SofaBaseMechanics
    SofaBaseTopology
    SofaBaseUtils
    SofaBaseVisual

    # SofaSimulation
    SofaSimulationCommon
    SofaSimulationGraph

    # SofaCommon
    SofaDeformable
    SofaEngine
    SofaExplicitOdeSolver
    SofaImplicitOdeSolver
    SofaLoader
    SofaMeshCollision
    SofaObjectInteraction
    SofaRigid
    SofaSimpleFem

    # SofaGeneral
    SofaBoundaryCondition
    SofaConstraint
    SofaGeneralAnimationLoop
    SofaGeneralDeformable
    SofaGeneralEngine
    SofaGeneralExplicitOdeSolver
    SofaGeneralImplicitOdeSolver
    SofaGeneralLinearSolver
    SofaGeneralLoader
    SofaGeneralMeshCollision
    SofaGeneralObjectInteraction
    SofaGeneralRigid
    SofaGeneralTopology
    SofaGeneralVisual
    SofaGraphComponent
    SofaTopologyMapping
    SofaUserInteraction

    # SofaAdvanced
    SofaNonUniformFem

    # SofaMisc
    SofaMiscCollision
    SofaMiscEngine
    SofaMiscExtra
    SofaMiscFem
    SofaMiscForceField
    SofaMiscMapping
    SofaMiscSolver
    SofaMiscTopology

    # Other (pluginized from SofaGeneral)
    SofaDenseSolver
    SofaExporter
    SofaHaptics
    SofaOpenglVisual
    SofaPreconditioner
    SofaSparseSolver
    SofaValidation
    )

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

set(SOFACOMPONENTALL_MODULES_INCLUDES "")
set(SOFACOMPONENTALL_MODULES_INITS "")
set(SOFACOMPONENTALL_MODULES_HAVES "")
foreach(module ${SOFACOMPONENTALL_MODULES})
    string(TOUPPER "${module}" module_upper)

    sofa_find_package(${module} QUIET)

    if(${module}_FOUND)
        target_link_libraries(${PROJECT_NAME} ${module})
        set(SOFACOMPONENTALL_MODULES_INCLUDES "${SOFACOMPONENTALL_MODULES_INCLUDES}\n#include <${module}/init${module}.h>")
        set(SOFACOMPONENTALL_MODULES_INITS "${SOFACOMPONENTALL_MODULES_INITS} \\\n    init${module}();")
    else()
        message("${PROJECT_NAME}: ${module} will not be in the collection.")
    endif()
    set(SOFACOMPONENTALL_MODULES_HAVES "${SOFACOMPONENTALL_MODULES_HAVES}\nset(SOFACOMPONENTALL_HAVE_${module_upper} ${SOFACOMPONENTALL_HAVE_${module_upper}})")
endforeach()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "SofaComponentAll"
    RELOCATABLE "collections"
    )
