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
    SofaEigen2Solver

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
    SofaGeneralSimpleFem
    SofaGeneralTopology
    SofaGeneralVisual
    SofaGraphComponent
    SofaTopologyMapping
    SofaUserInteraction

    # SofaAdvanced
    SofaNonUniformFem

    # SofaMisc
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

set(SOFACOMPONENTALL_MODULES_INCLUDES "")
set(SOFACOMPONENTALL_MODULES_INITS "")
set(SOFACOMPONENTALL_MODULES_HAVES "")
foreach(module ${SOFACOMPONENTALL_MODULES})
    string(TOUPPER "${module}" module_upper)

    sofa_find_package(${module})

    if(${module}_FOUND)
        set(SOFACOMPONENTALL_MODULES_INCLUDES "${SOFACOMPONENTALL_MODULES_INCLUDES}\n#include <${module}/init${module}.h>")
        set(SOFACOMPONENTALL_MODULES_INITS "${SOFACOMPONENTALL_MODULES_INITS} \\\n    init${module}();")
    endif()
    set(SOFACOMPONENTALL_MODULES_HAVES "${SOFACOMPONENTALL_MODULES_HAVES}\nset(SOFACOMPONENTALL_HAVE_${module_upper} ${SOFACOMPONENTALL_HAVE_${module_upper}})")
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
