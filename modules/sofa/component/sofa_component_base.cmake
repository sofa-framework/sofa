cmake_minimum_required(VERSION 2.8)

project("SofaComponentBase")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initComponentBase.h

    )
    
set(SOURCE_FILES

    initComponentBase.cpp

    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaEulerianFluid SofaSphFluid SofaVolumetricData SofaNonUniformFem SofaEigen3Solver )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_COMPONENT_BASE")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
