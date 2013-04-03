cmake_minimum_required(VERSION 2.8)

project("Modeler")

include(${SOFA_CMAKE_DIR}/pre.cmake)

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

set(LINKER_DEPENDENCIES SofaModeler SofaComponentMain )

include(${SOFA_CMAKE_DIR}/post.cmake)

