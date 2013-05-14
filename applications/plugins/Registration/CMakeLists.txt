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
	
if(SOFA_HAVE_IMAGE)
    
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.h")
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.inl")
    list(APPEND SOURCE_FILES "IntensityProfileRegistrationForceField.cpp")
    
endif()	

set(README_FILES

	registration.txt
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_REGISTRATION")
set(LINKER_DEPENDENCIES SofaGuiQt )

if(SOFA_HAVE_IMAGE)

    if(SOFA_IMAGE_HAVE_OPENCV)
    
        list(APPEND LINKER_DEPENDENCIES "ml")
        list(APPEND LINKER_DEPENDENCIES "cvaux")
        list(APPEND LINKER_DEPENDENCIES "highgui")
        list(APPEND LINKER_DEPENDENCIES "cv")
        list(APPEND LINKER_DEPENDENCIES "cxcore")

    endif()

endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)