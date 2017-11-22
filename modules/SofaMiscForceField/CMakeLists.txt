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
	    GearSpringForceField.h
        GearSpringForceField.inl             
        MatrixMass.h
        MatrixMass.inl
        MeshMatrixMass.h
        MeshMatrixMass.inl
        )
    list(APPEND SOURCE_FILES     
	    GearSpringForceField.cpp           
        MatrixMass.cpp
        MeshMatrixMass.cpp
        )

endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
    list(APPEND HEADER_FILES
	    ForceMaskOff.h  
        LineBendingSprings.h
        LineBendingSprings.inl
        WashingMachineForceField.h
        WashingMachineForceField.inl
		LennardJonesForceField.h
        LennardJonesForceField.inl
        )
    list(APPEND SOURCE_FILES
	    ForceMaskOff.cpp        
        LineBendingSprings.cpp
        WashingMachineForceField.cpp
		LennardJonesForceField.cpp
        )
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDeformable SofaBoundaryCondition SofaMiscTopology SofaGeneralTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC_FORCEFIELD")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
