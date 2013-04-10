cmake_minimum_required(VERSION 2.8)

project("SofaObjectInteraction")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initObjectInteraction.h 
    projectiveconstraintset/AttachConstraint.h 
    projectiveconstraintset/AttachConstraint.inl 
    interactionforcefield/BoxStiffSpringForceField.h 
    interactionforcefield/BoxStiffSpringForceField.inl 
    interactionforcefield/InteractionEllipsoidForceField.h 
    interactionforcefield/InteractionEllipsoidForceField.inl 
    interactionforcefield/PenalityContactForceField.h 
    interactionforcefield/PenalityContactForceField.inl 
    interactionforcefield/RepulsiveSpringForceField.h 
    interactionforcefield/RepulsiveSpringForceField.inl

    )
    
set(SOURCE_FILES

    initObjectInteraction.cpp 
    projectiveconstraintset/AttachConstraint.cpp 
    interactionforcefield/BoxStiffSpringForceField.cpp 
    interactionforcefield/InteractionEllipsoidForceField.cpp 
    interactionforcefield/PenalityContactForceField.cpp 
    interactionforcefield/RepulsiveSpringForceField.cpp
 
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_OBJECT_INTERACTION")
set(LINKER_DEPENDENCIES SofaDeformable)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
