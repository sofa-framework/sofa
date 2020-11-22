cmake_minimum_required(VERSION 3.12)
project(SofaDeformable)

find_package(SofaBase REQUIRED)

set(HEADER_FILES
    AngularSpringForceField.h
    AngularSpringForceField.inl
    MeshSpringForceField.h
    MeshSpringForceField.inl
    RestShapeSpringsForceField.h
    RestShapeSpringsForceField.inl
    PolynomialRestShapeSpringsForceField.h
    PolynomialRestShapeSpringsForceField.inl
    SpringForceField.h
    SpringForceField.inl
    StiffSpringForceField.h
    StiffSpringForceField.inl
    PolynomialSpringsForceField.h
    PolynomialSpringsForceField.inl
    config.h
    initDeformable.h
)

set(SOURCE_FILES
    AngularSpringForceField.cpp
    MeshSpringForceField.cpp
    RestShapeSpringsForceField.cpp
    PolynomialRestShapeSpringsForceField.cpp
    SpringForceField.cpp
    StiffSpringForceField.cpp
    PolynomialSpringsForceField.cpp
    initDeformable.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaCommon
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
