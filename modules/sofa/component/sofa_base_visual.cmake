cmake_minimum_required(VERSION 2.8)

project("SofaBaseVisual")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    initBaseVisual.h 
    visualmodel/BaseCamera.h 
    visualmodel/InteractiveCamera.h 
    visualmodel/RecordedCamera.h 
    visualmodel/VisualModelImpl.h 
    visualmodel/VisualStyle.h 
    visualmodel/VisualTransform.h

    )
    
set(SOURCE_FILES

    initBaseVisual.cpp 
    visualmodel/BaseCamera.cpp 
    visualmodel/InteractiveCamera.cpp 
    visualmodel/RecordedCamera.cpp 
    visualmodel/VisualModelImpl.cpp 
    visualmodel/VisualStyle.cpp 
    visualmodel/VisualTransform.cpp

    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILE_DEFINES "SOFA_BUILD_BASE_VISUAL")
set(LINK_DEPENDENCIES SofaBaseTopology)

include(${SOFA_CMAKE_DIR}/post.cmake)
