cmake_minimum_required(VERSION 2.8)

project("SofaEulerianFluid")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initEulerianFluid.h 
    behaviormodel/eulerianfluid/Fluid2D.h 
    behaviormodel/eulerianfluid/Fluid3D.h 
    behaviormodel/eulerianfluid/Grid2D.h 
    behaviormodel/eulerianfluid/Grid3D.h

    )
    
set(SOURCE_FILES

    initEulerianFluid.cpp 
    behaviormodel/eulerianfluid/Fluid2D.cpp 
    behaviormodel/eulerianfluid/Fluid3D.cpp 
    behaviormodel/eulerianfluid/Grid2D.cpp 
    behaviormodel/eulerianfluid/Grid3D.cpp

    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationTree )
    
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_EULERIAN_FLUID")
    
include(${SOFA_CMAKE_DIR}/post.cmake)
