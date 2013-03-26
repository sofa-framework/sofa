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

set(COMPILER_DEFINES "SOFA_BUILD_COMPONENT_BASE")
set(LINKER_DEPENDENCIES SofaEulerianFluid SofaSphFluid SofaVolumetricData SofaNonUniformFem SofaEigen2Solver)

include(${SOFA_CMAKE_DIR}/post.cmake)
