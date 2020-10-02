cmake_minimum_required(VERSION 3.1)
project(SofaBaseVisual)

set(HEADER_FILES
    BaseCamera.h
    InteractiveCamera.h
    VisualModelImpl.h
    VisualStyle.h
    config.h
    initBaseVisual.h
)

set(SOURCE_FILES
    BaseCamera.cpp
    InteractiveCamera.cpp
    VisualModelImpl.cpp
    VisualStyle.cpp
    initBaseVisual.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_VISUAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaBase ${PROJECT_NAME} ${PROJECT_NAME})
