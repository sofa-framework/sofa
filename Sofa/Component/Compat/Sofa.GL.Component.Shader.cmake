set(SOFAOPENGLVISUAL_SRC src/SofaOpenglVisual)

list(APPEND HEADER_FILES
    ${SOFAOPENGLVISUAL_SRC}/ClipPlane.h
    ${SOFAOPENGLVISUAL_SRC}/CompositingVisualLoop.h
    ${SOFAOPENGLVISUAL_SRC}/LightManager.h
    ${SOFAOPENGLVISUAL_SRC}/Light.h
    ${SOFAOPENGLVISUAL_SRC}/OrderIndependentTransparencyManager.h
    ${SOFAOPENGLVISUAL_SRC}/OglOITShader.h
    ${SOFAOPENGLVISUAL_SRC}/OglAttribute.h
    ${SOFAOPENGLVISUAL_SRC}/OglAttribute.inl
    ${SOFAOPENGLVISUAL_SRC}/OglShader.h
    ${SOFAOPENGLVISUAL_SRC}/OglShaderMacro.h
    ${SOFAOPENGLVISUAL_SRC}/OglShaderVisualModel.h
    ${SOFAOPENGLVISUAL_SRC}/OglShadowShader.h
    ${SOFAOPENGLVISUAL_SRC}/OglTexture.h
    ${SOFAOPENGLVISUAL_SRC}/OglTexturePointer.h
    ${SOFAOPENGLVISUAL_SRC}/OglVariable.h
    ${SOFAOPENGLVISUAL_SRC}/OglVariable.inl
    ${SOFAOPENGLVISUAL_SRC}/OglRenderingSRGB.h
    ${SOFAOPENGLVISUAL_SRC}/PostProcessManager.h
    ${SOFAOPENGLVISUAL_SRC}/VisualManagerPass.h
    ${SOFAOPENGLVISUAL_SRC}/VisualManagerSecondaryPass.h
)
