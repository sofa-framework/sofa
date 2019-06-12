cmake_minimum_required(VERSION 3.1)
project(SofaOpenglVisual VERSION 1.0 LANGUAGES CXX)

set(EXTRA_FILES
    README.md
    SofaOpenglVisualConfig.cmake.in
)

set(HEADER_FILES
    config.h
    initOpenGLVisual.h
    )

set(SOURCE_FILES
    initOpenGLVisual.cpp
    )

list(APPEND HEADER_FILES
    ClipPlane.h
    OglColorMap.h
    CompositingVisualLoop.h
    DataDisplay.h
    OglLabel.h
    OglModel.h
    PointSplatModel.h
    MergeVisualModels.h
    LightManager.h
    Light.h
    OrderIndependentTransparencyManager.h
    OglOITShader.h
    OglAttribute.h
    OglAttribute.inl
    OglShader.h
    OglShaderMacro.h
    OglShaderVisualModel.h
    OglShadowShader.h
    OglTexture.h
    OglTexturePointer.h
    OglVariable.h
    OglVariable.inl
    OglViewport.h
    OglCylinderModel.h
    OglGrid.h
    OglRenderingSRGB.h
    OglLineAxis.h
    OglSceneFrame.h
    PostProcessManager.h
    SlicedVolumetricModel.h
    VisualManagerPass.h
    VisualManagerSecondaryPass.h
    )

list(APPEND SOURCE_FILES
    ClipPlane.cpp
    OglColorMap.cpp
    CompositingVisualLoop.cpp
    DataDisplay.cpp
    OglGrid.cpp
    OglLabel.cpp
    OglModel.cpp
    PointSplatModel.cpp
    MergeVisualModels.cpp
    Light.cpp
    LightManager.cpp
    OrderIndependentTransparencyManager.cpp
    OglOITShader.cpp
    OglAttribute.cpp
    OglShader.cpp
    OglShaderMacro.cpp
    OglShaderVisualModel.cpp
    OglShadowShader.cpp
    OglTexture.cpp
    OglTexturePointer.cpp
    OglVariable.cpp
    OglViewport.cpp
    OglCylinderModel.cpp
    OglGrid.cpp
    OglRenderingSRGB.cpp
    OglLineAxis.cpp
    OglSceneFrame.cpp
    PostProcessManager.cpp
    SlicedVolumetricModel.cpp
    VisualManagerPass.cpp
    VisualManagerSecondaryPass.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseVisual SofaSimulationCommon)
target_compile_definitions(${PROJECT_NAME} PRIVATE COMPILE_FLAGS "-DSOFA_BUILD_SOFAOPENGLVISUAL")
target_compile_definitions(${PROJECT_NAME} PUBLIC COMPILE_FLAGS "-DSOFA_HAVE_SOFAOPENGLVISUAL")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")

sofa_generate_package(NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR ${PROJECT_NAME})
