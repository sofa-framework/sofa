// ShaderTextureMask::fragmentShaderText

PRECISON;
VARYING_FRAG vec2 texCoord;
uniform sampler2D textureUnit;
uniform sampler2D textureUnitMask;
FRAG_OUT_DEF;
void main()
{
	float m = texture2D(textureUnitMask,texCoord).r;
	if (m < 0.5)
		discard;
	FRAG_OUT=texture2D(textureUnit,texCoord)*m;
}
