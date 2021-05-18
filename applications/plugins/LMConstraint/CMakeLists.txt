cmake_minimum_required(VERSION 3.12)
project(LMConstraint VERSION 1.0 LANGUAGES CXX)

sofa_find_package(SofaFramework REQUIRED)
sofa_find_package(SofaBase REQUIRED)
sofa_find_package(SofaImplicitOdeSolver REQUIRED) 
sofa_find_package(SofaConstraint REQUIRED) 
sofa_find_package(SofaMiscCollision REQUIRED)
sofa_find_package(SofaDistanceGrid QUIET)

# List all files
set(LMCONSTRAINT_SRC_DIR src/LMConstraint)
set(HEADER_FILES
    ${LMCONSTRAINT_SRC_DIR}/config.h.in
    ${LMCONSTRAINT_SRC_DIR}/BaseLMConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/BarycentricDistanceLMConstraintContact.h
    ${LMCONSTRAINT_SRC_DIR}/BarycentricDistanceLMConstraintContact.inl
    ${LMCONSTRAINT_SRC_DIR}/LMConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/LMConstraint.inl
    ${LMCONSTRAINT_SRC_DIR}/ContactDescription.h
    ${LMCONSTRAINT_SRC_DIR}/DOFBlockerLMConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/DOFBlockerLMConstraint.inl
    ${LMCONSTRAINT_SRC_DIR}/FixedLMConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/FixedLMConstraint.inl
    ${LMCONSTRAINT_SRC_DIR}/DistanceLMContactConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/DistanceLMContactConstraint.inl
    ${LMCONSTRAINT_SRC_DIR}/DistanceLMConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/DistanceLMConstraint.inl
    ${LMCONSTRAINT_SRC_DIR}/LMConstraintSolver.h
    ${LMCONSTRAINT_SRC_DIR}/LMConstraintDirectSolver.h
    ${LMCONSTRAINT_SRC_DIR}/MechanicalWriteLMConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/PrecomputedLMConstraintCorrection.h
    ${LMCONSTRAINT_SRC_DIR}/PrecomputedLMConstraintCorrection.inl

    )

set(SOURCE_FILES
    ${LMCONSTRAINT_SRC_DIR}/initLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/BaseLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/BarycentricDistanceLMConstraintContact.cpp
    ${LMCONSTRAINT_SRC_DIR}/LMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/DOFBlockerLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/FixedLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/DistanceLMContactConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/DistanceLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/LMConstraintSolver.cpp
    ${LMCONSTRAINT_SRC_DIR}/LMConstraintDirectSolver.cpp
    ${LMCONSTRAINT_SRC_DIR}/MechanicalWriteLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/PrecomputedLMConstraintCorrection.cpp
    ${LMCONSTRAINT_SRC_DIR}/TetrahedronBarycentricDistanceLMConstraintContact.cpp
    )

if(SofaDistanceGrid_FOUND)
    list(APPEND SOURCE_FILES
        ${LMCONSTRAINT_SRC_DIR}/BarycentricDistanceLMConstraintContact_DistanceGrid.cpp     
    )
else()
    message(STATUS "SofaDistanceGrid not found: DistanceLM codes will not be compiled")
endif()


# Create the plugin library
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} )
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaBaseLinearSolver SofaImplicitOdeSolver SofaHelper SofaDefaultType SofaConstraint SofaMiscCollision)

if(SofaDistanceGrid_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDistanceGrid)
endif()

# Create package Config, Version & Target files.
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
