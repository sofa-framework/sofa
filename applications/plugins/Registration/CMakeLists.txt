cmake_minimum_required(VERSION 2.8)

project("SofaRegistration")

include(${SOFA_CMAKE_DIR}/pre.cmake)

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

    if(SOFA_IMAGE_HAVE_OPENCV)
    
        ##plugin external dependencies
        find_package(SofaOpenCVLibs REQUIRED)
        include_directories(${SOFAOPENCV_INCLUDE_DIR})

    endif()
    
    include_directories(${SOFAIMAGE_INCLUDE_DIRS})
    
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.h")
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.inl")
    list(APPEND SOURCE_FILES "IntensityProfileRegistrationForceField.cpp")
    
endif()	

set(README_FILE registration.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_REGISTRATION")
set(LINKER_DEPENDENCIES SofaGuiQt )

include(${SOFA_CMAKE_DIR}/post.cmake)