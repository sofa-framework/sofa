// ShaderSimpleTexture::fragmentShaderText

PRECISON;
VARYING_FRAG vec2 texCoord;
uniform sampler2D textureUnit;
FRAG_OUT_DEF;
void main()
{
	gl_FragColor=texture2D(textureUnit,texCoord);
}