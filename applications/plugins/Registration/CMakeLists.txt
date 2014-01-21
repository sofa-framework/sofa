include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    initRegistration.h 
	RegistrationContact.h 
	RegistrationContact.inl 
	RegistrationContactForceField.h 
	RegistrationContactForceField.inl 
	ClosestPointRegistrationForceField.h 
	ClosestPointRegistrationForceField.inl 
	RegistrationExporter.h

	)

set(SOURCE_FILES

	initRegistration.cpp 
	RegistrationContact.cpp 
	RegistrationContactForceField.cpp 
	ClosestPointRegistrationForceField.cpp 
	RegistrationExporter.cpp

	)
	
if(SOFA-PLUGIN_IMAGE)
    
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.h")
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.inl")
    list(APPEND SOURCE_FILES "IntensityProfileRegistrationForceField.cpp")
    
    include_directories("${SOFA_EXTLIBS_DIR}")
    include_directories("${SOFA_APPLICATIONS_PLUGINS_DIR}")

    AddLinkerDependencies(image)
endif()

set(README_FILES

	registration.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_REGISTRATION")
AddLinkerDependencies(SofaGuiQt )


include(${SOFA_CMAKE_DIR}/postProject.cmake)
