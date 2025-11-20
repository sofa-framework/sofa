cmake_minimum_required(VERSION 3.22)
project(MultiThreading VERSION 0.1)

set(HEADER_FILES
    src/MultiThreading/config.h.in
    src/MultiThreading/initMultiThreading.h
    src/MultiThreading/DataExchange.h
    src/MultiThreading/DataExchange.inl
    src/MultiThreading/MeanComputation.h
    src/MultiThreading/MeanComputation.inl
    src/MultiThreading/component/animationloop/AnimationLoopParallelScheduler.h
    src/MultiThreading/component/collision/detection/algorithm/ParallelBVHNarrowPhase.h
    src/MultiThreading/component/collision/detection/algorithm/ParallelBruteForceBroadPhase.h
    src/MultiThreading/component/linearsolver/iterative/ParallelCGLinearSolver.h
    src/MultiThreading/component/linearsolver/iterative/ParallelCGLinearSolver.inl
    src/MultiThreading/component/linearsolver/iterative/ParallelCompressedRowSparseMatrixMechanical.h
    src/MultiThreading/component/mapping/linear/BeamLinearMapping_mt.h
    src/MultiThreading/component/mapping/linear/BeamLinearMapping_mt.inl
    src/MultiThreading/component/mapping/linear/BeamLinearMapping_tasks.inl
    src/MultiThreading/component/solidmechanics/fem/elastic/ParallelHexahedronFEMForceField.h
    src/MultiThreading/component/solidmechanics/fem/elastic/ParallelHexahedronFEMForceField.inl
    src/MultiThreading/component/solidmechanics/fem/elastic/ParallelTetrahedronFEMForceField.h
    src/MultiThreading/component/solidmechanics/fem/elastic/ParallelTetrahedronFEMForceField.inl
    src/MultiThreading/component/solidmechanics/spring/ParallelStiffSpringForceField.h
    src/MultiThreading/component/solidmechanics/spring/ParallelStiffSpringForceField.inl
    src/MultiThreading/component/solidmechanics/spring/ParallelSpringForceField.h
    src/MultiThreading/component/solidmechanics/spring/ParallelSpringForceField.inl
    src/MultiThreading/component/solidmechanics/spring/ParallelMeshSpringForceField.h
    src/MultiThreading/component/solidmechanics/spring/ParallelMeshSpringForceField.inl
    src/MultiThreading/SceneCheckMultithreading.h
    src/MultiThreading/ParallelImplementationsRegistry.h
    )

set(SOURCE_FILES
    src/MultiThreading/initMultiThreading.cpp
    src/MultiThreading/DataExchange.cpp
    src/MultiThreading/MeanComputation.cpp
    src/MultiThreading/component/animationloop/AnimationLoopParallelScheduler.cpp
    src/MultiThreading/component/collision/detection/algorithm/ParallelBVHNarrowPhase.cpp
    src/MultiThreading/component/collision/detection/algorithm/ParallelBruteForceBroadPhase.cpp
    src/MultiThreading/component/linearsolver/iterative/ParallelCGLinearSolver.cpp
    src/MultiThreading/component/mapping/linear/BeamLinearMapping_mt.cpp
    src/MultiThreading/component/solidmechanics/fem/elastic/ParallelHexahedronFEMForceField.cpp
    src/MultiThreading/component/solidmechanics/fem/elastic/ParallelTetrahedronFEMForceField.cpp
    src/MultiThreading/component/solidmechanics/spring/ParallelSpringForceField.cpp
    src/MultiThreading/component/solidmechanics/spring/ParallelMeshSpringForceField.cpp
    src/MultiThreading/SceneCheckMultithreading.cpp
    src/MultiThreading/ParallelImplementationsRegistry.cpp
    )

set(DEPRECATED_DIR "compat/MultiThreading")
set(DEPRECATED_HEADER_FILES
    ${DEPRECATED_DIR}/TaskSchedulerUser.h
)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Simulation.Common REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
sofa_find_package(Sofa.Component.Mapping.Linear REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.Spring REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Iterative REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Simulation.Common)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Collision.Detection.Algorithm)
target_link_libraries(${PROJECT_NAME} Sofa.Component.SolidMechanics.FEM.Elastic)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Mapping.Linear)
target_link_libraries(${PROJECT_NAME} Sofa.Component.StateContainer)
target_link_libraries(${PROJECT_NAME} Sofa.Component.SolidMechanics.Spring)
target_link_libraries(${PROJECT_NAME} Sofa.Component.LinearSolver.Iterative)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/compat>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}_compat>
)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    RELOCATABLE "plugins"
    )

install(DIRECTORY compat/ DESTINATION include/${PROJECT_NAME}_compat COMPONENT headers)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(MULTITHREADING_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(MULTITHREADING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
