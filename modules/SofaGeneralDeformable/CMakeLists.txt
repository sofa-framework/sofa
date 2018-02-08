cmake_minimum_required(VERSION 3.1)
project(SofaGeneralDeformable)

set(HEADER_FILES
    config.h
    initGeneralDeformable.h
    )

set(SOURCE_FILES
    initGeneralDeformable.cpp
    )

list(APPEND HEADER_FILES
    FastTriangularBendingSprings.h
    FastTriangularBendingSprings.inl
    FrameSpringForceField.h
    FrameSpringForceField.inl
    QuadBendingSprings.h
    QuadBendingSprings.inl
    QuadularBendingSprings.h
    QuadularBendingSprings.inl
    RegularGridSpringForceField.h
    RegularGridSpringForceField.inl
    TriangleBendingSprings.h
    TriangleBendingSprings.inl
    TriangularBendingSprings.h
    TriangularBendingSprings.inl
    TriangularBiquadraticSpringsForceField.h
    TriangularBiquadraticSpringsForceField.inl
    TriangularQuadraticSpringsForceField.h
    TriangularQuadraticSpringsForceField.inl
    TriangularTensorMassForceField.h
    TriangularTensorMassForceField.inl
    VectorSpringForceField.h
    VectorSpringForceField.inl
    )
list(APPEND SOURCE_FILES
    FastTriangularBendingSprings.cpp
    FrameSpringForceField.cpp
    QuadBendingSprings.cpp
    QuadularBendingSprings.cpp
    RegularGridSpringForceField.cpp
    TriangleBendingSprings.cpp
    TriangularBendingSprings.cpp
    TriangularBiquadraticSpringsForceField.cpp
    TriangularQuadraticSpringsForceField.cpp
    TriangularTensorMassForceField.cpp
    VectorSpringForceField.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaDeformable)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_DEFORMABLE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
