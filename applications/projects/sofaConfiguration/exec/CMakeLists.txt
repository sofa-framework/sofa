cmake_minimum_required(VERSION 2.8)

project("SofaConfiguration")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    ../lib/SofaConfiguration.h 
    ../lib/ConfigurationParser.h  
    
    )

set(SOURCE_FILES

    Main.cpp 
    ../lib/SofaConfiguration.cpp 
    ../lib/ConfigurationParser.cpp   

    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

include(${SOFA_CMAKE_DIR}/post.cmake)

