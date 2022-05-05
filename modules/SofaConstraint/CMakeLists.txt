cmake_minimum_required(VERSION 3.12)
project(SofaConstraint)

set(SOFACONSTRAINT_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFACONSTRAINT_SRC}/config.h.in
    ${SOFACONSTRAINT_SRC}/initSofaConstraint.h
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.h
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.inl
    )

set(SOURCE_FILES
    ${SOFACONSTRAINT_SRC}/initSofaConstraint.cpp
    ${SOFACONSTRAINT_SRC}/ConstraintAttachBodyPerformer.cpp
    )

sofa_find_package(SofaBase REQUIRED) # SofaBaseLinearSolver
sofa_find_package(SofaImplicitOdeSolver REQUIRED)  
sofa_find_package(SofaMeshCollision REQUIRED)
sofa_find_package(SofaUserInteraction REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaImplicitOdeSolver SofaUserInteraction SofaBaseLinearSolver)

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Mapping.MappedMatrix, Sofa.Component.Constraint.Lagrangian.Model, Sofa.Component.Constraint.Lagrangian.Correction, Sofa.Component.Constraint.Lagrangian.Solver, Sofa.Component.AnimationLoop, Sofa.Component.Collision.Detection.Intersection and Sofa.Component.Collision.Response.Contact instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mapping.MappedMatrix REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Model REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Correction REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)
sofa_find_package(Sofa.Component.AnimationLoop REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping.MappedMatrix)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Model)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Correction)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Solver)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.AnimationLoop)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Intersection)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Contact)

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
