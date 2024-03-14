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

sofa_find_package(SofaBase REQUIRED) # SofaBaseLinearSolver
sofa_find_package(SofaImplicitOdeSolver REQUIRED)  
sofa_find_package(SofaUserInteraction REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaImplicitOdeSolver SofaUserInteraction SofaBaseLinearSolver)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Mapping.MappedMatrix, Sofa.Component.Constraint.Lagrangian.Model, Sofa.Component.Constraint.Lagrangian.Correction, Sofa.Component.Constraint.Lagrangian.Solver, Sofa.Component.AnimationLoop, Sofa.Component.Collision.Detection.Intersection, Sofa.Component.Collision.Response.Contact and Sofa.GUI.Component instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mapping.MappedMatrix REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Model REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Correction REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)
sofa_find_package(Sofa.Component.AnimationLoop REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
sofa_find_package(Sofa.GUI.Component REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping.MappedMatrix)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Model)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Correction)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian.Solver)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.AnimationLoop)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Intersection)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Contact)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Component)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
