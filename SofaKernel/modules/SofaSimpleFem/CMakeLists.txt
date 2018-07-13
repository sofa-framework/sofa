cmake_minimum_required(VERSION 3.1)
project(SofaSimpleFem)

set(HEADER_FILES
	CMHexahedronFEMForceField.h
	CMHexahedronFEMForceField.inl
    HexahedronFEMForceField.h
    HexahedronFEMForceField.inl
    TetrahedronFEMForceField.h
    TetrahedronFEMForceField.inl
    TetrahedronDiffusionFEMForceField.h
    TetrahedronDiffusionFEMForceField.inl
    config.h
    initSimpleFEM.h
)

set(SOURCE_FILES
    initSimpleFEM.cpp
	CMHexahedronFEMForceField.cpp
    HexahedronFEMForceField.cpp
    TetrahedronFEMForceField.cpp
    TetrahedronDiffusionFEMForceField.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIMPLE_FEM")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})
