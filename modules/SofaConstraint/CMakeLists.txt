cmake_minimum_required(VERSION 2.8)
project(SofaConstraint)

set(HEADER_FILES

    initConstraint.h
    LocalMinDistance.h
    LocalMinDistance.inl
    LMDNewProximityIntersection.h
    LMDNewProximityIntersection.inl
    FrictionContact.h
    FrictionContact.inl
    StickContactConstraint.h
    StickContactConstraint.inl
    UnilateralInteractionConstraint.h
    UnilateralInteractionConstraint.inl
    UncoupledConstraintCorrection.h
    UncoupledConstraintCorrection.inl
    PrecomputedConstraintCorrection.h
    PrecomputedConstraintCorrection.inl
    LinearSolverConstraintCorrection.h
    LinearSolverConstraintCorrection.inl
    GenericConstraintCorrection.h
    GenericConstraintCorrection.inl
    LCPConstraintSolver.h
    ConstraintSolverImpl.h
    FreeMotionAnimationLoop.h
    ContactDescription.h
    GenericConstraintSolver.h
    BilateralInteractionConstraint.h
    BilateralInteractionConstraint.inl
    ConstraintAnimationLoop.h
    StopperConstraint.h
    StopperConstraint.inl
    SlidingConstraint.h
    SlidingConstraint.inl
    ConstraintAttachBodyPerformer.h
    ConstraintAttachBodyPerformer.inl

    )

set(SOURCE_FILES

    initConstraint.cpp
    LocalMinDistance.cpp
    LMDNewProximityIntersection.cpp
    FrictionContact.cpp
    StickContactConstraint.cpp
    UnilateralInteractionConstraint.cpp
    UncoupledConstraintCorrection.cpp
    PrecomputedConstraintCorrection.cpp
    LinearSolverConstraintCorrection.cpp
    LCPConstraintSolver.cpp
    ConstraintSolverImpl.cpp
    FreeMotionAnimationLoop.cpp
    GenericConstraintSolver.cpp
    BilateralInteractionConstraint.cpp
    ConstraintAnimationLoop.cpp
    StopperConstraint.cpp
    SlidingConstraint.cpp
    ConstraintAttachBodyPerformer.cpp
    GenericConstraintCorrection.cpp

    )

if(SOFA_HAVE_EIGEN2)
 	list(APPEND HEADER_FILES "BarycentricDistanceLMConstraintContact.h")
	list(APPEND HEADER_FILES "BarycentricDistanceLMConstraintContact.inl")
	list(APPEND HEADER_FILES "DOFBlockerLMConstraint.h")
	list(APPEND HEADER_FILES "DOFBlockerLMConstraint.inl")
	list(APPEND HEADER_FILES "FixedLMConstraint.h")
	list(APPEND HEADER_FILES "FixedLMConstraint.inl")
	list(APPEND HEADER_FILES "DistanceLMContactConstraint.h")
	list(APPEND HEADER_FILES "DistanceLMContactConstraint.inl")
	list(APPEND HEADER_FILES "DistanceLMConstraint.h")
	list(APPEND HEADER_FILES "DistanceLMConstraint.inl")
	list(APPEND HEADER_FILES "LMConstraintSolver.h")
	list(APPEND HEADER_FILES "LMConstraintDirectSolver.h")

	list(APPEND SOURCE_FILES "BarycentricDistanceLMConstraintContact.cpp")
	list(APPEND SOURCE_FILES "DOFBlockerLMConstraint.cpp")
	list(APPEND SOURCE_FILES "FixedLMConstraint.cpp")
	list(APPEND SOURCE_FILES "DistanceLMContactConstraint.cpp")
	list(APPEND SOURCE_FILES "DistanceLMConstraint.cpp")
	list(APPEND SOURCE_FILES "LMConstraintSolver.cpp")
	list(APPEND SOURCE_FILES "LMConstraintDirectSolver.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaMeshCollision SofaSimpleFem SofaImplicitOdeSolver SofaUserInteraction SofaBaseLinearSolver)
if(SOFA_HAVE_EIGEN2)
   target_link_libraries( ${PROJECT_NAME} SofaEigen2Solver)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFACONSTRAINT_COMPILER_FLAGS "-DSOFA_BUILD_CONSTRAINT")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFACONSTRAINT_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaConstraint_libraries
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"

)

