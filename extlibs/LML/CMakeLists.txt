include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    Direction.h
    Force.h 
    ForceUnit.h 
    Load.h 
    Loads.h 
	LoadsVersion.h 
    Pressure.h 
    PressureUnit.h 
    Rotation.h 
    RotationUnit.h 
    TargetList.h 
    Translation.h 
    TranslationUnit.h 
    Unit.h 
    ValueEvent.h 
	XMLLoads.h 
	xmlio.h
 
	)

set(SOURCE_FILES

    Force.cpp 
    ForceUnit.cpp 
    Load.cpp 
    Loads.cpp 
    PressureUnit.cpp 
    RotationUnit.cpp 
    TargetList.cpp 
    Translation.cpp 
    TranslationUnit.cpp 
    ValueEvent.cpp 
	XMLLoads.cpp 

	)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

# fix link errors when linked with shared libraries
if(${CMAKE_COMPILER_IS_GNUCC})
    AddCompilerFlags("-fPIC")
endif()

AddCompilerDefinitions("use_namespace")

include(${SOFA_CMAKE_DIR}/postProject.cmake)

