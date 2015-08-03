include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initObjectInteraction.h
    AttachConstraint.h
    AttachConstraint.inl
    BoxStiffSpringForceField.h
    BoxStiffSpringForceField.inl
    InteractionEllipsoidForceField.h
    InteractionEllipsoidForceField.inl
    PenalityContactForceField.h
    PenalityContactForceField.inl
    RepulsiveSpringForceField.h
    RepulsiveSpringForceField.inl
    )

set(SOURCE_FILES

    initObjectInteraction.cpp
    AttachConstraint.cpp
    BoxStiffSpringForceField.cpp
    InteractionEllipsoidForceField.cpp
    PenalityContactForceField.cpp
    RepulsiveSpringForceField.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_OBJECT_INTERACTION")
AddLinkerDependencies(SofaDeformable SofaMiscMapping)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
