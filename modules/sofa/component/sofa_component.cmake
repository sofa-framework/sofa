cmake_minimum_required(VERSION 2.8)

project("SofaComponent")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    init.h

    )
    
set(SOURCE_FILES

    init.cpp

    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_COMPONENT")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
