include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initOpenGLVisual.h
    OglModel.h
    OglViewport.h
    Light.h
    LightManager.h
    PointSplatModel.h
    OglRenderingSRGB.h
    ClipPlane.h
    CompositingVisualLoop.h
    ColorMap.h
    DataDisplay.h
    OglLabel.h
    OglCylinderModel.h
    OglSceneFrame.h
    OglGrid.h
    OglLineAxis.h

    )

set(SOURCE_FILES

    initOpenGLVisual.cpp
    OglModel.cpp
    OglViewport.cpp
    Light.cpp
    LightManager.cpp
    PointSplatModel.cpp
    OglRenderingSRGB.cpp
    ClipPlane.cpp
    CompositingVisualLoop.cpp
    ColorMap.cpp
    DataDisplay.cpp
    OglLabel.cpp
    OglCylinderModel.cpp
    OglSceneFrame.cpp
    OglGrid.cpp
    OglLineAxis.cpp

    )

if(SOFA-EXTERNAL_GLEW)
	list(APPEND HEADER_FILES "OglAttribute.h")
	list(APPEND HEADER_FILES "OglAttribute.inl")
	list(APPEND HEADER_FILES "OglShader.h")
	list(APPEND HEADER_FILES "OglShaderMacro.h")
	list(APPEND HEADER_FILES "OglShaderVisualModel.h")
	list(APPEND HEADER_FILES "OglShadowShader.h")
	list(APPEND HEADER_FILES "OglTetrahedralModel.h")
	list(APPEND HEADER_FILES "OglTetrahedralModel.inl")
	list(APPEND HEADER_FILES "OglTexture.h")
	list(APPEND HEADER_FILES "OglVariable.h")
	list(APPEND HEADER_FILES "OglVariable.inl")
	list(APPEND HEADER_FILES "PostProcessManager.h")
	list(APPEND HEADER_FILES "SlicedVolumetricModel.h")
	list(APPEND HEADER_FILES "VisualManagerPass.h")
	list(APPEND HEADER_FILES "VisualManagerSecondaryPass.h")

	list(APPEND SOURCE_FILES "OglAttribute.cpp")
	list(APPEND SOURCE_FILES "OglShader.cpp")
	list(APPEND SOURCE_FILES "OglShaderMacro.cpp")
	list(APPEND SOURCE_FILES "OglShaderVisualModel.cpp")
	list(APPEND SOURCE_FILES "OglShadowShader.cpp")
	list(APPEND SOURCE_FILES "OglTetrahedralModel.cpp")
	list(APPEND SOURCE_FILES "OglTexture.cpp")
	list(APPEND SOURCE_FILES "OglVariable.cpp")
	list(APPEND SOURCE_FILES "PostProcessManager.cpp")
	list(APPEND SOURCE_FILES "SlicedVolumetricModel.cpp")
	list(APPEND SOURCE_FILES "VisualManagerPass.cpp")
	list(APPEND SOURCE_FILES "VisualManagerSecondaryPass.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_OPENGL_VISUAL")
AddLinkerDependencies(SofaBaseVisual SofaSimulationCommon)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
