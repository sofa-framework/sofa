include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../lib/SofaConfiguration.h 
    ../lib/ConfigurationParser.h  
    
    )

set(SOURCE_FILES

    Main.cpp 
    ../lib/SofaConfiguration.cpp 
    ../lib/ConfigurationParser.cpp   

    )

if(APPLE)

    set(RC_FILES "sofaConfiguration.icns")

else()

    set(RC_FILES "sofa.rc")

endif()

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${RC_FILES})

if(WIN32)

    AddLinkerDependencies("shell32")

endif()

if(APPLE)

    AddLinkerDependencies("CoreFoundation")
    
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

