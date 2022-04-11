cmake_minimum_required(VERSION 3.12)
project(SofaConstraint)

set(SOFACONSTRAINT_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFACONSTRAINT_SRC}/config.h.in
    ${SOFACONSTRAINT_SRC}/initSofaConstraint.h
    )

set(SOURCE_FILES
    ${SOFACONSTRAINT_SRC}/initSofaConstraint.cpp
    )

list(APPEND HEADER_FILES
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.h
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.inl
    ${SOFACONSTRAINT_SRC}/ContactIdentifier.h
    ${SOFACONSTRAINT_SRC}/FrictionContact.h
    ${SOFACONSTRAINT_SRC}/FrictionContact.inl
    ${SOFACONSTRAINT_SRC}/LocalMinDistance.h
    ${SOFACONSTRAINT_SRC}/StickContactConstraint.h
    ${SOFACONSTRAINT_SRC}/StickContactConstraint.inl
    )
list(APPEND SOURCE_FILES
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.cpp
    ${SOFACONSTRAINT_SRC}/ContactIdentifier.cpp
    ${SOFACONSTRAINT_SRC}/FrictionContact.cpp
    ${SOFACONSTRAINT_SRC}/LocalMinDistance.cpp
    ${SOFACONSTRAINT_SRC}/StickContactConstraint.cpp
    )

sofa_find_package(SofaBase REQUIRED) # SofaBaseLinearSolver
sofa_find_package(SofaImplicitOdeSolver REQUIRED)  
sofa_find_package(SofaMeshCollision REQUIRED)
sofa_find_package(SofaUserInteraction REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaImplicitOdeSolver SofaUserInteraction SofaBaseLinearSolver)

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Mapping.MappedMatrix, Sofa.Component.Constraint.Lagrangian.Model, Sofa.Component.Constraint.Lagrangian.Correction, Sofa.Component.Constraint.Lagrangian.Solver and Sofa.Component.AnimationLoop instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mapping.MappedMatrix REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Model REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Correction REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)
sofa_find_package(Sofa.Component.AnimationLoop REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping.MappedMatrix)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Model)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Correction)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Solver)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.AnimationLoop)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFACONSTRAINT_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFACONSTRAINT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
