// ShaderSimpleTexture::fragmentShaderText

PRECISON;
VARYING_FRAG vec2 texCoord;
uniform sampler2D textureUnit;
FRAG_OUT_DEF;
void main()
{
	FRAG_OUT=texture2D(textureUnit,texCoord);
}