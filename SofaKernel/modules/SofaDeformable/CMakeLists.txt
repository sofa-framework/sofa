cmake_minimum_required(VERSION 3.1)
project(SofaDeformable)

set(HEADER_FILES
    AngularSpringForceField.h
    AngularSpringForceField.inl
    MeshSpringForceField.h
    MeshSpringForceField.inl
    RestShapeSpringsForceField.h
    RestShapeSpringsForceField.inl
    SpringForceField.h
    SpringForceField.inl
    StiffSpringForceField.h
    StiffSpringForceField.inl
    config.h
    initDeformable.h
)

set(SOURCE_FILES
    AngularSpringForceField.cpp
    MeshSpringForceField.cpp
    RestShapeSpringsForceField.cpp
    SpringForceField.cpp
    StiffSpringForceField.cpp
    initDeformable.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_DEFORMABLE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
