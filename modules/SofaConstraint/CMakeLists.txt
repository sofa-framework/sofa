include(${SOFA_CMAKE_DIR}/preProject.cmake)

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

if(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)
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

AddCompilerDefinitions("SOFA_BUILD_CONSTRAINT" )
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaMeshCollision SofaSimpleFem SofaImplicitOdeSolver SofaUserInteraction )
AddSourceDependencies(eigen)
if(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)
    AddLinkerDependencies(SofaEigen2Solver)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
