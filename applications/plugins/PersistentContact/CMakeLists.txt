cmake_minimum_required(VERSION 3.22)
project(PersistentContact VERSION 0.1)


set(HEADER_FILES
    src/PersistentContact/config.h.in
    src/PersistentContact/PersistentContactBarycentricMapping.h
    src/PersistentContact/PersistentContactBarycentricMapping.inl
    src/PersistentContact/PersistentContactMapping.h
    src/PersistentContact/PersistentContactRigidMapping.h
    src/PersistentContact/PersistentContactRigidMapping.inl
    src/PersistentContact/PersistentFrictionContact.h
    src/PersistentContact/PersistentFrictionContact.inl
    src/PersistentContact/PersistentUnilateralInteractionConstraint.h
    src/PersistentContact/PersistentUnilateralInteractionConstraint.inl
)

set(SOURCE_FILES
    src/PersistentContact/PersistentContactBarycentricMapping.cpp
    src/PersistentContact/PersistentContactMapping.cpp
    src/PersistentContact/PersistentContactRigidMapping.cpp
    src/PersistentContact/PersistentFrictionContact.cpp
    src/PersistentContact/PersistentUnilateralInteractionConstraint.cpp
    src/PersistentContact/initPersistentContact.cpp
)

set(README_FILES PersistentContact.txt)

set(README_FILES
	PersistentContact.txt)

sofa_find_package(Sofa.Component.Mapping REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Model REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME}
    Sofa.Component.Mapping
    Sofa.Component.Collision.Response.Contact
    Sofa.Component.Constraint.Lagrangian.Model
)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
