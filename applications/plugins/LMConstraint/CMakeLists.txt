cmake_minimum_required(VERSION 3.12)
project(LMConstraint VERSION 1.0 LANGUAGES CXX)

find_package(SofaFramework REQUIRED)
find_package(SofaBase REQUIRED)
find_package(SofaCommon REQUIRED) 
find_package(SofaConstraint REQUIRED) 

# List all files
set(LMCONSTRAINT_SRC_DIR src/LMConstraint)
set(HEADER_FILES
    ${LMCONSTRAINT_SRC_DIR}/config.h.in
    ${LMCONSTRAINT_SRC_DIR}/behavior/BaseLMConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/behavior/LMConstraint.h
    ${LMCONSTRAINT_SRC_DIR}/behavior/LMConstraint.inl
    ${LMCONSTRAINT_SRC_DIR}/BarycentricDistanceLMConstraintContact.h
    ${LMCONSTRAINT_SRC_DIR}/BarycentricDistanceLMConstraintContact.inl
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
set(HEADER_FILES_TO_MOC
    )
set(SOURCE_FILES
    ${LMCONSTRAINT_SRC_DIR}/initLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/behavior/BaseLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/behavior/LMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/TetrahedronBarycentricDistanceLMConstraintContact.cpp
    ${LMCONSTRAINT_SRC_DIR}/BarycentricDistanceLMConstraintContact_DistanceGrid.cpp
    ${LMCONSTRAINT_SRC_DIR}/BarycentricDistanceLMConstraintContact.cpp
    ${LMCONSTRAINT_SRC_DIR}/DOFBlockerLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/FixedLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/DistanceLMContactConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/DistanceLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/LMConstraintSolver.cpp
    ${LMCONSTRAINT_SRC_DIR}/LMConstraintDirectSolver.cpp
    ${LMCONSTRAINT_SRC_DIR}/MechanicalWriteLMConstraint.cpp
    ${LMCONSTRAINT_SRC_DIR}/PrecomputedLMConstraintCorrection.cpp
    )


# Create the plugin library
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES_TO_MOC} ${MOCCED_HEADER_FILES} ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaHelper SofaDefaultType SofaConstraint)


# Create package Config, Version & Target files.
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(PLUGINEXAMPLE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(PLUGINEXAMPLE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(PluginExample_test)
endif()
