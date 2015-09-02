cmake_minimum_required(VERSION 2.8)
project(PersistentContact)


set(HEADER_FILES
    config.h
    PersistentContactBarycentricMapping.h
    PersistentContactBarycentricMapping.inl
    PersistentContactMapping.h
    PersistentContactRigidMapping.h
    PersistentContactRigidMapping.inl
    PersistentFrictionContact.h
    PersistentFrictionContact.inl
    PersistentUnilateralInteractionConstraint.h
    PersistentUnilateralInteractionConstraint.inl
)

set(SOURCE_FILES
    PersistentContactBarycentricMapping.cpp
    PersistentContactMapping.cpp
    PersistentContactRigidMapping.cpp
    PersistentFrictionContact.cpp
    PersistentUnilateralInteractionConstraint.cpp
    initPersistentContact.cpp
)

set(README_FILES PersistentContact.txt)

set(README_FILES
	PersistentContact.txt)

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PERSISTENTCONTACT")
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaRigid SofaConstraint)

install(TARGETS ${PROJECT_NAME}
    COMPONENT PersistentContact_libraries
    EXPORT PersistentContactTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
