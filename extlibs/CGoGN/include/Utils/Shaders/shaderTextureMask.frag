// ShaderTextureMask::fragmentShaderText

PRECISION;
VARYING_FRAG vec2 texCoord;
uniform sampler2D textureUnit;
uniform sampler2D textureUnitMask;
FRAG_OUT_DEF;
void main()
{
	float m = TEXTURE2D(textureUnitMask,texCoord).r;
	if (m < 0.5)
		discard;
	FRAG_OUT=TEXTURE2D(textureUnit,texCoord)*m;
}
