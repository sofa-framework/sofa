cmake_minimum_required(VERSION 3.1)
project(SofaBaseMechanics)

set(HEADER_FILES
    AddMToMatrixFunctor.h
    BarycentricMapping.h
    BarycentricMapping.inl
    DiagonalMass.h
    DiagonalMass.inl
    IdentityMapping.h
    IdentityMapping.inl
    MappedObject.h
    MappedObject.inl
    MechanicalObject.h
    MechanicalObject.inl
    SubsetMapping.h
    SubsetMapping.inl
    UniformMass.h
    UniformMass.inl
    config.h
    initBaseMechanics.h
)

set(SOURCE_FILES
    BarycentricMapping.cpp
    DiagonalMass.cpp
    IdentityMapping.cpp
    MappedObject.cpp
    MechanicalObject.cpp
    SubsetMapping.cpp
    UniformMass.cpp
    initBaseMechanics.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_MECHANICS")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaBase ${PROJECT_NAME} ${PROJECT_NAME})
