cmake_minimum_required(VERSION 3.1)
project(SofaMiscCollision)

message(STATUS "SofaMiscCollision:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initMiscCollision.h
)

set(SOURCE_FILES
    initMiscCollision.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        BarycentricStickContact.h
        BarycentricStickContact.inl
        DefaultCollisionGroupManager.h
        RuleBasedContactManager.h
        SolverMerger.h
        SpatialGridPointModel.h
        TetrahedronDiscreteIntersection.h
        TetrahedronModel.h
        TriangleModelInRegularGrid.h
        )
    list(APPEND SOURCE_FILES
        BarycentricStickContact.cpp
        DefaultCollisionGroupManager.cpp
        RuleBasedContactManager.cpp
        SolverMerger.cpp
        SpatialGridPointModel.cpp
        TetrahedronBarycentricPenalityContact.cpp
        TetrahedronDiscreteIntersection.cpp
        TetrahedronFrictionContact.cpp
        TetrahedronModel.cpp
        TetrahedronRayContact.cpp
        TriangleModelInRegularGrid.cpp
        TetrahedronBarycentricDistanceLMConstraintContact.cpp
        )

    if(SOFA_HAVE_SMP)
        list(APPEND HEADER_FILES ParallelCollisionPipeline.h)
        list(APPEND SOURCE_FILES ParallelCollisionPipeline.cpp)
    endif()

    if(NOT SOFA_NO_OPENGL)
        list(APPEND HEADER_FILES RayTriangleVisitor.h)
        list(APPEND SOURCE_FILES RayTriangleVisitor.cpp)
    endif()

    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")
endif()



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaGeneralMeshCollision SofaGeneralDeformable SofaConstraint SofaExplicitOdeSolver SofaGeneralExplicitOdeSolver SofaImplicitOdeSolver SofaSphFluid)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaEigen2Solver)
if(NOT SOFA_NO_OPENGL)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaOpenglVisual)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_COLLISION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
