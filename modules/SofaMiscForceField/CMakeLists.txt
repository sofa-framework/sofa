cmake_minimum_required(VERSION 3.1)
project(SofaMiscForceField)


message(STATUS "SofaMiscForceField:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initMiscForcefield.h
)

set(SOURCE_FILES
    initMiscForcefield.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        ForceMaskOff.h
        GearSpringForceField.h
        GearSpringForceField.inl
        LennardJonesForceField.h
        LennardJonesForceField.inl
        LineBendingSprings.h
        LineBendingSprings.inl
        MatrixMass.h
        MatrixMass.inl
        MeshMatrixMass.h
        MeshMatrixMass.inl
        WashingMachineForceField.h
        WashingMachineForceField.inl
        )
    list(APPEND SOURCE_FILES
        ForceMaskOff.cpp
        GearSpringForceField.cpp
        LennardJonesForceField.cpp
        LineBendingSprings.cpp
        MatrixMass.cpp
        MeshMatrixMass.cpp
        WashingMachineForceField.cpp
        )
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
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDeformable SofaBoundaryCondition SofaMiscTopology SofaGeneralTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_FORCEFIELD")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
