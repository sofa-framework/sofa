include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initOpenGLVisual.h 
    ../visualmodel/OglModel.h 
    # ../visualmodel/OglViewport.h 
    # ../visualmodel/Light.h 
    # ../visualmodel/LightManager.h 
    # ../visualmodel/PointSplatModel.h 
    # ../visualmodel/OglRenderingSRGB.h 
    # ../visualmodel/ClipPlane.h 
    # ../visualmodel/CompositingVisualLoop.h 
    ../visualmodel/ColorMap.h 
    # ../visualmodel/DataDisplay.h
    # ../visualmodel/OglLabel.h
    # ../visualmodel/OglCylinderModel.h
    # ../visualmodel/OglSceneFrame.h
    # ../visualmodel/OglGrid.h
    # ../visualmodel/OglLineAxis.h

    )
    
set(SOURCE_FILES

    ../initOpenGLVisual.cpp 
    ../visualmodel/OglModel.cpp 
    # ../visualmodel/OglViewport.cpp 
    # ../visualmodel/Light.cpp 
    # ../visualmodel/LightManager.cpp 
    # ../visualmodel/PointSplatModel.cpp 
    # ../visualmodel/OglRenderingSRGB.cpp 
    # ../visualmodel/ClipPlane.cpp 
    # ../visualmodel/CompositingVisualLoop.cpp 
    ../visualmodel/ColorMap.cpp 
    # ../visualmodel/DataDisplay.cpp
    # ../visualmodel/OglLabel.cpp
    # ../visualmodel/OglCylinderModel.cpp
    # ../visualmodel/OglSceneFrame.cpp
    # ../visualmodel/OglGrid.cpp
    # ../visualmodel/OglLineAxis.cpp

    )
    
# if(SOFA-EXTERNAL_GLEW)
# 	list(APPEND HEADER_FILES "../visualmodel/OglAttribute.h")
# 	list(APPEND HEADER_FILES "../visualmodel/OglAttribute.inl")
# 	list(APPEND HEADER_FILES "../visualmodel/OglShader.h")
# 	list(APPEND HEADER_FILES "../visualmodel/OglShaderMacro.h")
# 	list(APPEND HEADER_FILES "../visualmodel/OglShaderVisualModel.h")
# 	list(APPEND HEADER_FILES "../visualmodel/OglShadowShader.h")
# 	list(APPEND HEADER_FILES "../visualmodel/OglTetrahedralModel.h")
# 	list(APPEND HEADER_FILES "../visualmodel/OglTetrahedralModel.inl")
# 	list(APPEND HEADER_FILES "../visualmodel/OglTexture.h")
# 	list(APPEND HEADER_FILES "../visualmodel/OglVariable.h")
# 	list(APPEND HEADER_FILES "../visualmodel/OglVariable.inl")
# 	list(APPEND HEADER_FILES "../visualmodel/PostProcessManager.h")
# 	list(APPEND HEADER_FILES "../visualmodel/SlicedVolumetricModel.h")
# 	list(APPEND HEADER_FILES "../visualmodel/VisualManagerPass.h")
# 	list(APPEND HEADER_FILES "../visualmodel/VisualManagerSecondaryPass.h")

# 	list(APPEND SOURCE_FILES "../visualmodel/OglAttribute.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/OglShader.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/OglShaderMacro.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/OglShaderVisualModel.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/OglShadowShader.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/OglTetrahedralModel.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/OglTexture.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/OglVariable.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/PostProcessManager.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/SlicedVolumetricModel.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/VisualManagerPass.cpp")
# 	list(APPEND SOURCE_FILES "../visualmodel/VisualManagerSecondaryPass.cpp")
# endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_OPENGL_VISUAL")
AddLinkerDependencies(SofaBaseVisual SofaSimulationCommon)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
