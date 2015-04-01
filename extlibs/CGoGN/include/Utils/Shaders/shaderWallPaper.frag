// ShaderWallPaper::fragmentShaderText

PRECISION;
VARYING_FRAG vec2 texCoord;
uniform sampler2D textureUnit;
FRAG_OUT_DEF;
void main()
{
	FRAG_OUT=TEXTURE2D(textureUnit,texCoord);
}
