cmake_minimum_required(VERSION 2.8)

project("SofaComponentAdvanced")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initComponentAdvanced.h
    
    )
    
set(SOURCE_FILES

    initComponentAdvanced.cpp

    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaEulerianFluid SofaSphFluid SofaVolumetricData SofaNonUnionFem SofaEigen2Solver )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_COMPONENT_ADVANCED")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
