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
    src/SofaOpenglVisual/ClipPlane.h
    src/SofaOpenglVisual/OglColorMap.h
    src/SofaOpenglVisual/CompositingVisualLoop.h
    src/SofaOpenglVisual/DataDisplay.h
    src/SofaOpenglVisual/OglLabel.h
    src/SofaOpenglVisual/OglModel.h
    src/SofaOpenglVisual/PointSplatModel.h
    src/SofaOpenglVisual/MergeVisualModels.h
    src/SofaOpenglVisual/LightManager.h
    src/SofaOpenglVisual/Light.h
    src/SofaOpenglVisual/OrderIndependentTransparencyManager.h
    src/SofaOpenglVisual/OglOITShader.h
    src/SofaOpenglVisual/OglAttribute.h
    src/SofaOpenglVisual/OglAttribute.inl
    src/SofaOpenglVisual/OglShader.h
    src/SofaOpenglVisual/OglShaderMacro.h
    src/SofaOpenglVisual/OglShaderVisualModel.h
    src/SofaOpenglVisual/OglShadowShader.h
    src/SofaOpenglVisual/OglTexture.h
    src/SofaOpenglVisual/OglTexturePointer.h
    src/SofaOpenglVisual/OglVariable.h
    src/SofaOpenglVisual/OglVariable.inl
    src/SofaOpenglVisual/OglViewport.h
    src/SofaOpenglVisual/OglCylinderModel.h
    src/SofaOpenglVisual/OglGrid.h
    src/SofaOpenglVisual/OglRenderingSRGB.h
    src/SofaOpenglVisual/OglLineAxis.h
    src/SofaOpenglVisual/OglSceneFrame.h
    src/SofaOpenglVisual/PostProcessManager.h
    src/SofaOpenglVisual/SlicedVolumetricModel.h
    src/SofaOpenglVisual/VisualManagerPass.h
    src/SofaOpenglVisual/VisualManagerSecondaryPass.h
    )

list(APPEND SOURCE_FILES
    src/SofaOpenglVisual/ClipPlane.cpp
    src/SofaOpenglVisual/OglColorMap.cpp
    src/SofaOpenglVisual/CompositingVisualLoop.cpp
    src/SofaOpenglVisual/DataDisplay.cpp
    src/SofaOpenglVisual/OglGrid.cpp
    src/SofaOpenglVisual/OglLabel.cpp
    src/SofaOpenglVisual/OglModel.cpp
    src/SofaOpenglVisual/PointSplatModel.cpp
    src/SofaOpenglVisual/MergeVisualModels.cpp
    src/SofaOpenglVisual/Light.cpp
    src/SofaOpenglVisual/LightManager.cpp
    src/SofaOpenglVisual/OrderIndependentTransparencyManager.cpp
    src/SofaOpenglVisual/OglOITShader.cpp
    src/SofaOpenglVisual/OglAttribute.cpp
    src/SofaOpenglVisual/OglShader.cpp
    src/SofaOpenglVisual/OglShaderMacro.cpp
    src/SofaOpenglVisual/OglShaderVisualModel.cpp
    src/SofaOpenglVisual/OglShadowShader.cpp
    src/SofaOpenglVisual/OglTexture.cpp
    src/SofaOpenglVisual/OglTexturePointer.cpp
    src/SofaOpenglVisual/OglVariable.cpp
    src/SofaOpenglVisual/OglViewport.cpp
    src/SofaOpenglVisual/OglCylinderModel.cpp
    src/SofaOpenglVisual/OglGrid.cpp
    src/SofaOpenglVisual/OglRenderingSRGB.cpp
    src/SofaOpenglVisual/OglLineAxis.cpp
    src/SofaOpenglVisual/OglSceneFrame.cpp
    src/SofaOpenglVisual/PostProcessManager.cpp
    src/SofaOpenglVisual/SlicedVolumetricModel.cpp
    src/SofaOpenglVisual/VisualManagerPass.cpp
    src/SofaOpenglVisual/VisualManagerSecondaryPass.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseVisual SofaSimulationCommon)
target_compile_definitions(${PROJECT_NAME} PRIVATE COMPILE_FLAGS "-DSOFA_BUILD_SOFAOPENGLVISUAL")
target_compile_definitions(${PROJECT_NAME} PUBLIC COMPILE_FLAGS "-DSOFA_HAVE_SOFAOPENGLVISUAL")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

sofa_generate_package(NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR ${PROJECT_NAME})
