include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    Main.cpp
    )
    
UseQt()
if(APPLE)

    set(RC_FILES "../Modeler.icns")
    
else()
    
    set(RC_FILES "sofa.rc")

endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(SofaModeler SofaComponentMain )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

