cmake_minimum_required(VERSION 2.8)
project(PersistentContact)

set(PERSISTENTCONTACT_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(PERSISTENTCONTACT_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(PERSISTENTCONTACT_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    PersistentContact.h
	PersistentContactMapping.h
	PersistentFrictionContact.h
	PersistentFrictionContact.inl
	PersistentUnilateralInteractionConstraint.h
	PersistentUnilateralInteractionConstraint.inl
	PersistentContactBarycentricMapping.h
	PersistentContactBarycentricMapping.inl
	PersistentContactRigidMapping.h
	PersistentContactRigidMapping.inl)

set(SOURCE_FILES
    PersistentContactMapping.cpp
	PersistentFrictionContact.cpp
	PersistentUnilateralInteractionConstraint.cpp
	PersistentContactBarycentricMapping.cpp
	PersistentContactRigidMapping.cpp
    initPersistentContact.cpp)

set(README_FILES
	PersistentContact.txt)

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PERSISTENTCONTACT")
target_link_libraries(${PROJECT_NAME} SofaBaseMechanics SofaRigid SofaConstraint)

install(TARGETS ${PROJECT_NAME}
    COMPONENT PersistentContact_libraries
    EXPORT PersistentContactTargets
    RUNTIME DESTINATION "${PERSISTENTCONTACT_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${PERSISTENTCONTACT_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${PERSISTENTCONTACT_INSTALL_LIB_DIR}"
    PUBLIC_HEADER DESTINATION "${PERSISTENTCONTACT_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
