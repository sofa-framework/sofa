include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    Main.cpp
    )
    
set(RC_FILES

    sofa.rc
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

include_directories(${SOFA_APPLICATIONS_PLUGINS_DIR})
AddLinkerDependencies(SofaGuiMain SofaOpenCL)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

