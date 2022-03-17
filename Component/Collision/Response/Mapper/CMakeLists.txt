cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Collision.Response.Mapper LANGUAGES CXX)

set(response/SOFACOMPONENTCOLLISIONDETECTIONRESPONSEMAPPER_SOURCE_DIR "src/sofa/component/collision/response/mapper")

set(HEADER_FILES
    ${response/SOFACOMPONENTCOLLISIONDETECTIONRESPONSEMAPPER_SOURCE_DIR}/config.h.in
    ${response/SOFACOMPONENTCOLLISIONDETECTIONRESPONSEMAPPER_SOURCE_DIR}/init.h
    # ${SOFABASECOLLISION_SRC}/BaseContactMapper.h
    # ${SOFAMESHCOLLISION_SRC}/BarycentricContactMapper.h
    # ${SOFAMESHCOLLISION_SRC}/BarycentricContactMapper.inl
    # ${SOFAMESHCOLLISION_SRC}/BarycentricPenalityContact.h
    # ${SOFAMESHCOLLISION_SRC}/BarycentricPenalityContact.inl
    # ${SOFABASECOLLISION_SRC}/DefaultContactManager.cpp
    # ${SOFAMESHCOLLISION_SRC}/IdentityContactMapper.h
    # ${SOFAMESHCOLLISION_SRC}/IdentityContactMapper.inl
    # ${SOFAMESHCOLLISION_SRC}/RigidContactMapper.h
    # ${SOFAMESHCOLLISION_SRC}/RigidContactMapper.inl
    # ${SOFAMESHCOLLISION_SRC}/SubsetContactMapper.h
    # ${SOFAMESHCOLLISION_SRC}/SubsetContactMapper.inl
    # ${SOFACONSTRAINT_SRC}/ContactIdentifier.h
    # ${SOFACONSTRAINT_SRC}/FrictionContact.h
    # ${SOFACONSTRAINT_SRC}/FrictionContact.inl
    # ${SOFACONSTRAINT_SRC}/StickContactConstraint.h
    # ${SOFACONSTRAINT_SRC}/StickContactConstraint.inl
    # ${SOFAUSERINTERACTION_SRC}/RayContact.h
)

set(SOURCE_FILES
    ${response/SOFACOMPONENTCOLLISIONDETECTIONRESPONSEMAPPER_SOURCE_DIR}/init.cpp
    # ${SOFABASECOLLISION_SRC}/BaseContactMapper.cpp
    # ${SOFAMESHCOLLISION_SRC}/BarycentricContactMapper.cpp
    # ${SOFAMESHCOLLISION_SRC}/BarycentricPenalityContact.cpp
    # ${SOFABASECOLLISION_SRC}/DefaultContactManager.cpp
    # ${SOFAMESHCOLLISION_SRC}/IdentityContactMapper.cpp
    # ${SOFAMESHCOLLISION_SRC}/RigidContactMapper.cpp
    # ${SOFAMESHCOLLISION_SRC}/SubsetContactMapper.cpp
    # ${SOFACONSTRAINT_SRC}/ContactIdentifier.cpp
    # ${SOFACONSTRAINT_SRC}/FrictionContact.cpp
    # ${SOFACONSTRAINT_SRC}/StickContactConstraint.cpp
    # ${SOFAUSERINTERACTION_SRC}/RayContact.cpp
)


sofa_find_package(Sofa.SimulationCore REQUIRED)
# sofa_find_package(Sofa.Component.Mapping)
# sofa_find_package(Sofa.Component.Container) # BarycentricMapper needs MechanicalObject
# sofa_find_package(Sofa.Component.SolidMechanics.Spring) # BarycentricPenalityContact
# sofa_find_package(Sofa.Component.Constraint.Lagrangian) # FrictionContact needs UnilateralCC
# sofa_find_package(Sofa.Component.Collision.Model)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${WRAPPER_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.SimulationCore)
# target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping)
# target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Container)
# target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.Spring)
# target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Constraint.Lagrangian)
# target_link_libraries(${PROJECT_NAME} PUBLIC ofa.Component.Collision.Model)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
