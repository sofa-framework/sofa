cmake_minimum_required(VERSION 2.8.12)
project(SofaBaseVisual)

set(HEADER_FILES
    BaseCamera.h
    InteractiveCamera.h
    RecordedCamera.h
    VisualModelImpl.h
    VisualStyle.h
    VisualTransform.h
    config.h
    initBaseVisual.h
)

set(SOURCE_FILES
    BaseCamera.cpp
    InteractiveCamera.cpp
    RecordedCamera.cpp
    VisualModelImpl.cpp
    VisualStyle.cpp
    VisualTransform.cpp
    initBaseVisual.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_VISUAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaBase_install(${PROJECT_NAME})
