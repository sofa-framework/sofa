cmake_minimum_required(VERSION 3.12)
project(SofaBoundaryCondition LANGUAGES CXX)

set(SOFABOUNDARYCONDITION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABOUNDARYCONDITION_SRC}/config.h.in
    ${SOFABOUNDARYCONDITION_SRC}/initSofaBoundaryCondition.h
    )

set(SOURCE_FILES
    ${SOFABOUNDARYCONDITION_SRC}/initSofaBoundaryCondition.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBaseLinearSolver)

message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. You may use Sofa.Component.Constraint.Projective, Sofa.Component.MechanicalLoad and Sofa.Component.SolidMechanics.FEM.Damping instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Constraint.Projective REQUIRED)
sofa_find_package(Sofa.Component.MechanicalLoad REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Damping REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Projective)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.MechanicalLoad)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.FEM.Damping)

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
cmake_dependent_option(SOFABOUNDARYCONDITION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABOUNDARYCONDITION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
