// ShaderTextureDepth::fragmentShaderText

PRECISON;
VARYING_FRAG vec2 texCoord;
uniform sampler2D textureUnit;
uniform sampler2D textureDepthUnit;
FRAG_OUT_DEF;
void main()
{
    gl_FragDepth = texture2D(textureDepthUnit,texCoord).r;
	gl_FragColor = texture2D(textureUnit,texCoord);
}