include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
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
    RepulsiveSpringForceField.inl)

set(SOURCE_FILES
    initObjectInteraction.cpp
    AttachConstraint.cpp
    BoxStiffSpringForceField.cpp
    InteractionEllipsoidForceField.cpp
    PenalityContactForceField.cpp
    RepulsiveSpringForceField.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_OBJECT_INTERACTION")
AddLinkerDependencies(Deformable)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

