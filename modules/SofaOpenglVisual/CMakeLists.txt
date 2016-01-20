cmake_minimum_required(VERSION 2.8.12)
project(SofaOpenglVisual)

set(HEADER_FILES
    ClipPlane.h
    ColorMap.h
    CompositingVisualLoop.h
    DataDisplay.h
    Light.h
    LightManager.h
    OglCylinderModel.h
    OglGrid.h
    OglLabel.h
    OglLineAxis.h
    OglModel.h
    OglRenderingSRGB.h
    OglSceneFrame.h
    PointSplatModel.h
    config.h
    initOpenGLVisual.h
    MergeVisualModels.h
)

set(SOURCE_FILES
    ClipPlane.cpp
    ColorMap.cpp
    CompositingVisualLoop.cpp
    DataDisplay.cpp
    Light.cpp
    LightManager.cpp
    OglCylinderModel.cpp
    OglGrid.cpp
    OglLabel.cpp
    OglLineAxis.cpp
    OglModel.cpp
    OglRenderingSRGB.cpp
    OglSceneFrame.cpp
    PointSplatModel.cpp
    initOpenGLVisual.cpp
    MergeVisualModels.cpp
)

if(SOFA_HAVE_GLEW)

    list(APPEND HEADER_FILES
        OglAttribute.h
        OglAttribute.inl
        OglShader.h
        OglShaderMacro.h
        OglShaderVisualModel.h
        OglShadowShader.h
        OglTetrahedralModel.h
        OglTetrahedralModel.inl
        OglTexture.h
        OglTexturePointer.h
        OglVariable.h
        OglVariable.inl
        OglViewport.h
        PostProcessManager.h
        SlicedVolumetricModel.h
        VisualManagerPass.h
        VisualManagerSecondaryPass.h
    )

    list(APPEND SOURCE_FILES
        OglAttribute.cpp
        OglShader.cpp
        OglShaderMacro.cpp
        OglShaderVisualModel.cpp
        OglShadowShader.cpp
        OglTetrahedralModel.cpp
        OglTexture.cpp
        OglTexturePointer.cpp
        OglVariable.cpp
        OglViewport.cpp
        PostProcessManager.cpp
        SlicedVolumetricModel.cpp
        VisualManagerPass.cpp
        VisualManagerSecondaryPass.cpp
    )

endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseVisual SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_OPENGL_VISUAL")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
