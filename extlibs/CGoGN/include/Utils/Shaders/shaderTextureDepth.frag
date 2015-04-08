// ShaderTextureDepth::fragmentShaderText

PRECISION;
VARYING_FRAG vec2 texCoord;
uniform sampler2D textureUnit;
uniform sampler2D textureDepthUnit;
FRAG_OUT_DEF;
void main()
{
	gl_FragDepth = TEXTURE2D(textureDepthUnit,texCoord).r;
	FRAG_OUT = TEXTURE2D(textureUnit,texCoord);
}
