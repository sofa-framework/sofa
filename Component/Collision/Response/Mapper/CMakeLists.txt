cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Collision.Response.Mapper LANGUAGES CXX)

set(SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR "src/sofa/component/collision/response/mapper")

set(HEADER_FILES
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/init.h
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/BaseContactMapper.h
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/BarycentricContactMapper.h
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/BarycentricContactMapper.inl
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/IdentityContactMapper.h
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/IdentityContactMapper.inl
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/RigidContactMapper.h
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/RigidContactMapper.inl
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/SubsetContactMapper.h
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/SubsetContactMapper.inl
)

set(SOURCE_FILES
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/BaseContactMapper.cpp
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/BarycentricContactMapper.cpp
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/IdentityContactMapper.cpp
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/RigidContactMapper.cpp
    ${SOFACOMPONENTCOLLISIONRESPONSEMAPPER_SOURCE_DIR}/SubsetContactMapper.cpp
)

sofa_find_package(Sofa.SimulationCore REQUIRED)
sofa_find_package(Sofa.Component.Collision.Model REQUIRED)
sofa_find_package(Sofa.Component.Mapping.Linear REQUIRED)
sofa_find_package(Sofa.Component.Mapping.NonLinear REQUIRED)
# sofa_find_package(Sofa.Component.StateContainer REQUIRED) # BarycentricMapper needs MechanicalObject

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.SimulationCore)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Model)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping.Linear)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping.NonLinear)
# target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.StateContainer)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
