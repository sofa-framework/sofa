cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initBaseVisual.h 
    ../visualmodel/BaseCamera.h 
    ../visualmodel/InteractiveCamera.h 
    ../visualmodel/RecordedCamera.h 
    ../visualmodel/VisualModelImpl.h 
    ../visualmodel/VisualStyle.h 
    ../visualmodel/VisualTransform.h

    )
    
set(SOURCE_FILES

    ../initBaseVisual.cpp 
    ../visualmodel/BaseCamera.cpp 
    ../visualmodel/InteractiveCamera.cpp 
    ../visualmodel/RecordedCamera.cpp 
    ../visualmodel/VisualModelImpl.cpp 
    ../visualmodel/VisualStyle.cpp 
    ../visualmodel/VisualTransform.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_BASE_VISUAL")
set(LINKER_DEPENDENCIES SofaBaseTopology)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
