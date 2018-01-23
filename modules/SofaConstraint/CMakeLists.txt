cmake_minimum_required(VERSION 3.1)
project(SofaConstraint)

################################ COMPONENTS HERE ARE THE NG-SET ####################################

set(HEADER_FILES
    config.h
    initConstraint.h
)

set(SOURCE_FILES
    initConstraint.cpp
)

############################## COMPONENTS HERE ARE THE LIGHT-SET ###################################
if(SOFA_BUILD_COMPONENTSET_LIGHT)
    list(APPEND HEADER_FILES

        )
    list(APPEND SOURCE_FILES

        )
endif()

############################## COMPONENTS HERE ARE THE STANDARD-SET ################################
if(SOFA_BUILD_COMPONENTSET_STANDARD)
    list(APPEND HEADER_FILES
        BilateralConstraintResolution.h
        BilateralInteractionConstraint.h
        BilateralInteractionConstraint.inl
        ConstraintAnimationLoop.h
        ConstraintAttachBodyPerformer.h
        ConstraintAttachBodyPerformer.inl
        ConstraintSolverImpl.h
        ContactDescription.h
        ContactIdentifier.h
        FreeMotionAnimationLoop.h
        FrictionContact.h
        FrictionContact.inl
        GenericConstraintCorrection.h
        GenericConstraintSolver.h
        LCPConstraintSolver.h
        LMDNewProximityIntersection.h
        LMDNewProximityIntersection.inl
        LinearSolverConstraintCorrection.h
        LinearSolverConstraintCorrection.inl
        LocalMinDistance.h
        LocalMinDistance.inl
        PrecomputedConstraintCorrection.h
        PrecomputedConstraintCorrection.inl
        SlidingConstraint.h
        SlidingConstraint.inl
        StickContactConstraint.h
        StickContactConstraint.inl
        StopperConstraint.h
        StopperConstraint.inl
        UncoupledConstraintCorrection.h
        UncoupledConstraintCorrection.inl
        UnilateralInteractionConstraint.h
        UnilateralInteractionConstraint.inl
        BarycentricDistanceLMConstraintContact.h
        BarycentricDistanceLMConstraintContact.inl
        DOFBlockerLMConstraint.h
        DOFBlockerLMConstraint.inl
        FixedLMConstraint.h
        FixedLMConstraint.inl
        DistanceLMContactConstraint.h
        DistanceLMContactConstraint.inl
        DistanceLMConstraint.h
        DistanceLMConstraint.inl
        LMConstraintSolver.h
        LMConstraintDirectSolver.h
        )
    list(APPEND SOURCE_FILES
        BilateralInteractionConstraint.cpp
        ConstraintAnimationLoop.cpp
        ConstraintAttachBodyPerformer.cpp
        ConstraintSolverImpl.cpp
        ContactIdentifier.cpp
        FreeMotionAnimationLoop.cpp
        FrictionContact.cpp
        GenericConstraintCorrection.cpp
        GenericConstraintSolver.cpp
        LCPConstraintSolver.cpp
        LMDNewProximityIntersection.cpp
        LinearSolverConstraintCorrection.cpp
        LocalMinDistance.cpp
        PrecomputedConstraintCorrection.cpp
        SlidingConstraint.cpp
        StickContactConstraint.cpp
        StopperConstraint.cpp
        UncoupledConstraintCorrection.cpp
        UnilateralInteractionConstraint.cpp
        BarycentricDistanceLMConstraintContact.cpp
        DOFBlockerLMConstraint.cpp
        FixedLMConstraint.cpp
        DistanceLMContactConstraint.cpp
        DistanceLMConstraint.cpp
        LMConstraintSolver.cpp
        LMConstraintDirectSolver.cpp
        )
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)

endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaSimpleFem SofaImplicitOdeSolver SofaUserInteraction SofaBaseLinearSolver)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_CONSTRAINT")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
