cmake_minimum_required(VERSION 3.1)
project(SofaOpenglVisual)

if(${SOFA_NO_OPENGL})
    message(SEND_ERROR "Warning: CMake flag SOFA_NO_OPENGL is active.")
    message("Flag SOFA_NO_OPENGL incompatible with the SofaOpenglVisual module.")    
endif()

set(HEADER_FILES
    ClipPlane.h
    OglColorMap.h
    CompositingVisualLoop.h
    DataDisplay.h
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
    OglColorMap.cpp
    CompositingVisualLoop.cpp
    DataDisplay.cpp
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
        PostProcessManager.h
        SlicedVolumetricModel.h
        VisualManagerPass.h
        VisualManagerSecondaryPass.h
    )

    list(APPEND SOURCE_FILES
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
